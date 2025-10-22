# Filename: plot_alert_direct.py
# Description:
#   Compute per-plot direct deforestation and land-use metrics by intersecting:
#     - a deforestation raster (pixel-class based)
#     - protected areas polygons
#     - farming areas polygons
#   The function parallelizes over plots and shares the raster efficiently across workers.
#
# Public API:
#   - alert_direct(...)
#
# Author: Steven Sotelo
#
# Notes:
#   - Robust to Windows limitations by falling back from SharedMemory to np.memmap.
#   - Avoids rasterio 'boundless' kwarg (uses window intersection instead).
#   - All areas reported in hectares; proportions in [0, 1].
#   - CRS: everything is reprojected to the raster CRS; if a vector has no CRS, it is assumed
#     to already be in the raster CRS.
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import math
import tempfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.windows import transform as window_transform
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from shapely.ops import unary_union
from shapely import wkb
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Globals used inside worker processes (set by _init_worker)
# --------------------------------------------------------------------------------------

_A_SHM = None               # SharedMemory handle (if used)
_A_MEMMAP_PATH = None       # memmap path on disk (fallback if SHM fails)
_A_ARRAY: Optional[np.ndarray] = None  # Read-only view of raster (SHM or memmap)
_A_SHAPE: Optional[Tuple[int, int]] = None
_A_DTYPE: Optional[np.dtype] = None
_A_RASTER_PATH = None   # streaming mode: path to the raster on disk

_A_TRANSFORM = None         # Affine transform of the raster
_A_WIDTH: Optional[int] = None
_A_HEIGHT: Optional[int] = None

_A_PIXEL_AREA_HA: Optional[float] = None  # area of a pixel in hectares
_A_DEFO_VALUE: Optional[int] = None       # deforestation class value to count

# Protected/Farming unions (single multipolygon per layer for fast area intersections)
_G_PROTECTED_UNION = None
_G_FARMING_UNION = None


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    """Return num/den with protection for zero/NaN denominators."""
    # If denominator is <= 0 (or very close) return 0 safely
    if den is None or den <= 0:
        return 0.0
    return float(num) / float(den)


def _bounds_to_window(bounds, full_transform, raster_width: int, raster_height: int) -> Tuple[int, int, int, int]:
    """Convert geometry bounds to a clamped raster window (no 'boundless' kwarg).

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) of the geometry in raster CRS.
    full_transform : Affine
        Raster affine transform.
    raster_width : int
        Number of columns in the raster.
    raster_height : int
        Number of rows in the raster.

    Returns
    -------
    row_off : int
    col_off : int
    h : int
    w : int

    Notes
    -----
    - Uses 'from_bounds' to compute a tentative window and then intersects it with
      the full image window to clamp to the raster extent.
    - If geometry lies completely outside, (h, w) will be (0, 0).
    """
    # 1) tentative window from bounds
    win = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], transform=full_transform)

    # 2) intersect with full image
    full_win = Window(col_off=0, row_off=0, width=raster_width, height=raster_height)
    win = win.intersection(full_win)

    # 3) clamp and cast
    row_off = int(max(0, math.floor(win.row_off)))
    col_off = int(max(0, math.floor(win.col_off)))
    h = int(max(0, math.ceil(win.height)))
    w = int(max(0, math.ceil(win.width)))

    # edge clamp
    if row_off >= raster_height or col_off >= raster_width:
        return 0, 0, 0, 0
    h = min(h, raster_height - row_off)
    w = min(w, raster_width - col_off)
    return row_off, col_off, h, w


# --------------------------------------------------------------------------------------
# Worker-side initialization and per-plot processing
# --------------------------------------------------------------------------------------

def _init_worker(
    shape: Tuple[int, int],
    dtype: str,
    transform,
    width: int,
    height: int,
    pixel_area_ha: float,
    defo_value: int,
    shm_name: Optional[str] = None,
    memmap_path: Optional[str] = None,
    protected_union_wkb: Optional[bytes] = None,
    farming_union_wkb: Optional[bytes] = None,
    raster_path: Optional[str] = None,   # <-- nuevo
):
    """Initializer executed once in each worker process."""
    global _A_SHM, _A_MEMMAP_PATH, _A_ARRAY, _A_SHAPE, _A_DTYPE
    global _A_TRANSFORM, _A_WIDTH, _A_HEIGHT, _A_PIXEL_AREA_HA, _A_DEFO_VALUE
    global _G_PROTECTED_UNION, _G_FARMING_UNION, _A_RASTER_PATH

    _A_SHAPE = tuple(shape)
    _A_DTYPE = np.dtype(dtype)
    _A_TRANSFORM = transform
    _A_WIDTH = int(width)
    _A_HEIGHT = int(height)
    _A_PIXEL_AREA_HA = float(pixel_area_ha)
    _A_DEFO_VALUE = int(defo_value)

    # Attach raster backing
    if shm_name is not None:
        from multiprocessing import shared_memory
        _A_SHM = shared_memory.SharedMemory(name=shm_name)
        _A_ARRAY = np.ndarray(_A_SHAPE, dtype=_A_DTYPE, buffer=_A_SHM.buf)
        _A_RASTER_PATH = None
    elif memmap_path is not None:
        _A_MEMMAP_PATH = memmap_path
        _A_ARRAY = np.memmap(_A_MEMMAP_PATH, dtype=_A_DTYPE, mode="r", shape=_A_SHAPE)
        _A_RASTER_PATH = None
    elif raster_path is not None:
        # Streaming mode: no array in memory; read windows from file per call
        _A_ARRAY = None
        _A_RASTER_PATH = raster_path
    else:
        raise RuntimeError("Worker received neither SHM nor memmap nor raster_path (streaming).")

    # Unions
    _G_PROTECTED_UNION = wkb.loads(protected_union_wkb) if protected_union_wkb else None
    _G_FARMING_UNION = wkb.loads(farming_union_wkb) if farming_union_wkb else None


def _intersect_raster_deforestation(geom) -> float:
    global _A_ARRAY, _A_TRANSFORM, _A_WIDTH, _A_HEIGHT, _A_DEFO_VALUE, _A_PIXEL_AREA_HA, _A_RASTER_PATH

    # Compute window for the geometry envelope
    row_off, col_off, h, w = _bounds_to_window(geom.bounds, _A_TRANSFORM, _A_WIDTH, _A_HEIGHT)
    if h == 0 or w == 0:
        return 0.0

    # Window transform
    win = Window(col_off, row_off, w, h)
    win_tf = window_transform(win, _A_TRANSFORM)

    # Build mask True inside geometry
    mask_inside = geometry_mask(
        [mapping(geom)],
        out_shape=(h, w),
        transform=win_tf,
        invert=True,
        all_touched=False,
    )

    if _A_ARRAY is not None:
        # SHM / memmap fast path
        view = _A_ARRAY[row_off: row_off + h, col_off: col_off + w]
    else:
        # Streaming mode: open the raster file and read only the window
        with rasterio.open(_A_RASTER_PATH) as src:
            view = src.read(1, window=win)

    hits = (view == _A_DEFO_VALUE) & mask_inside
    pixels = int(np.count_nonzero(hits))
    return float(pixels) * float(_A_PIXEL_AREA_HA)


def _intersect_area_ha(geom, union_geom) -> float:
    """Return intersection area in hectares between 'geom' and a union geometry.

    If 'union_geom' is None or empty, returns 0.0.
    """
    if (union_geom is None) or union_geom.is_empty or geom.is_empty:
        return 0.0
    return float(geom.intersection(union_geom).area) / 10000.0


def _process_one(plot_id: str, geom) -> Dict[str, float]:
    """Compute all metrics for a single plot geometry.

    Returns
    -------
    dict with:
      - id
      - plot_area
      - deforested_area
      - deforested_proportion
      - protected_areas_area
      - protected_areas_proportion
      - farming_in_area
      - farming_in_proportion
      - farming_out_area
      - farming_out_proportion
      - alert_direct (bool as Python bool)
    """
    # Area of the plot in hectares (assuming CRS in meters)
    plot_area_ha = float(geom.area) / 10000.0

    # Deforestation (ha) by intersecting raster class
    defo_ha = _intersect_raster_deforestation(geom)

    # Protected areas (ha) via geometry intersection
    prot_ha = _intersect_area_ha(geom, _G_PROTECTED_UNION)

    # Farming in (ha) and out (ha) via geometry intersection
    farm_in_ha = _intersect_area_ha(geom, _G_FARMING_UNION)
    farm_out_ha = max(plot_area_ha - farm_in_ha, 0.0)

    # Proportions
    defo_prop = _safe_div(defo_ha, plot_area_ha)
    prot_prop = _safe_div(prot_ha, plot_area_ha)
    farm_in_prop = _safe_div(farm_in_ha, plot_area_ha)
    farm_out_prop = _safe_div(farm_out_ha, plot_area_ha)

    # Alert flag if any deforestation detected
    alert = bool(defo_ha > 0.0)

    return {
        "id": plot_id,
        "plot_area": plot_area_ha,
        "deforested_area": defo_ha,
        "deforested_proportion": defo_prop,
        "protected_areas_area": prot_ha,
        "protected_areas_proportion": prot_prop,
        "farming_in_area": farm_in_ha,
        "farming_in_proportion": farm_in_prop,
        "farming_out_area": farm_out_ha,
        "farming_out_proportion": farm_out_prop,
        "alert_direct": alert,
    }


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

def alert_direct(
    plots: gpd.GeoDataFrame,
    deforestation: str,
    protected_areas: str,
    farming_areas: str,
    deforestation_value: int = 2,
    n_workers: int = 2,
    id_column: str = "id",
) -> pd.DataFrame:
    """Compute per-plot direct deforestation and land-use metrics.

    Parameters
    ----------
    plots : gpd.GeoDataFrame
        Plot polygons. Must contain an ID column (default 'id').
    deforestation : str
        Path to a raster file (e.g., GeoTIFF) containing a deforestation class code.
    protected_areas : str
        Path to a polygon vector dataset (shp/geojson) for protected areas.
    farming_areas : str
        Path to a polygon vector dataset (shp/geojson) for farming areas.
    deforestation_value : int, default 2
        Pixel value that encodes "deforestation" in the raster.
    n_workers : int, default 2
        Number of worker processes (>=1). If 1, computation is serial.
    id_column : str, default "id"
        Name of the plot ID column in `plots`.

    Returns
    -------
    pd.DataFrame
        Columns:
          - id
          - plot_area
          - deforested_area
          - deforested_proportion
          - protected_areas_area
          - protected_areas_proportion
          - farming_in_area
          - farming_in_proportion
          - farming_out_area
          - farming_out_proportion
          - alert_direct (bool)
    """
    if id_column not in plots.columns:
        raise ValueError(f"plots is missing the ID column '{id_column}'")

    # ------------------------
    # Load raster (once)
    # ------------------------
    with rasterio.open(deforestation) as src:
        raster_arr = src.read(1)  # (H, W)
        transform = src.transform
        width, height = src.width, src.height
        raster_crs = src.crs

        # Pixel area in m^2 -> to hectares
        # |det(Affine)| = pixel size area in projected units (assume meters)
        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
        pixel_area_ha = float(pixel_area_m2) / 10000.0

    # ------------------------
    # Load vectors (once)
    # ------------------------
    # Read protected and farming layers
    prot = gpd.read_file(protected_areas)
    farm = gpd.read_file(farming_areas)

    # If a vector has no CRS, assume it is already in raster CRS
    if raster_crs is not None:
        if prot.crs is None:
            prot = prot.set_crs(raster_crs, inplace=False)
        if farm.crs is None:
            farm = farm.set_crs(raster_crs, inplace=False)
    # Reproject to raster CRS if available
    if raster_crs is not None:
        if prot.crs != raster_crs:
            prot = prot.to_crs(raster_crs)
        if farm.crs != raster_crs:
            farm = farm.to_crs(raster_crs)

    # Prepare unions for fast intersection (can be None if empty)
    prot_union = unary_union(prot.geometry) if (len(prot) > 0 and prot.geometry.notnull().any()) else None
    farm_union = unary_union(farm.geometry) if (len(farm) > 0 and farm.geometry.notnull().any()) else None

    # Encode unions as WKB for safe pickling to workers
    prot_union_wkb = wkb.dumps(prot_union) if prot_union else None
    farm_union_wkb = wkb.dumps(farm_union) if farm_union else None

    # ------------------------
    # Prepare plots (reproject & clean)
    # ------------------------
    plots = plots[[id_column, "geometry"]].copy()

    # If plots has no CRS and raster has CRS, we assume they are in the raster CRS
    if raster_crs is not None and plots.crs is None:
        plots = plots.set_crs(raster_crs, inplace=False)

    # Reproject plots to raster CRS (if available)
    if raster_crs is not None and plots.crs != raster_crs:
        plots = plots.to_crs(raster_crs)

    # Drop null geometries
    plots = plots[plots.geometry.notnull()].reset_index(drop=True)

    # ------------------------
    # Parallel or serial path
    # ------------------------
    results: List[Dict] = []

    if n_workers <= 1 or len(plots) == 0:
        # --- Serial path: set globals directly (no SHM/memmap needed) ---
        global _A_ARRAY, _A_SHAPE, _A_DTYPE
        global _A_TRANSFORM, _A_WIDTH, _A_HEIGHT, _A_PIXEL_AREA_HA, _A_DEFO_VALUE
        global _G_PROTECTED_UNION, _G_FARMING_UNION

        _A_ARRAY = raster_arr                      # usamos el array en memoria del proceso actual
        _A_SHAPE = raster_arr.shape
        _A_DTYPE = raster_arr.dtype
        _A_TRANSFORM = transform
        _A_WIDTH = width
        _A_HEIGHT = height
        _A_PIXEL_AREA_HA = float(pixel_area_ha)
        _A_DEFO_VALUE = int(deforestation_value)

        # Asignar las uniones directamente (no WKB en el proceso principal)
        _G_PROTECTED_UNION = prot_union
        _G_FARMING_UNION = farm_union

        # Iterate plots one by one (serial)
        for _, row in tqdm(plots.iterrows(), total=len(plots), desc="Computing direct alerts (serial)"):
            metrics = _process_one(plot_id=str(row[id_column]), geom=row.geometry)
            results.append(metrics)

    else:
        # Parallel path: try SHM -> memmap -> streaming
        use_shm = False
        use_memmap = False
        shm = None
        memmap_path = None
        streaming = False

        try:
            from multiprocessing import shared_memory
            shm = shared_memory.SharedMemory(create=True, size=raster_arr.nbytes)
            shm_view = np.ndarray(raster_arr.shape, dtype=raster_arr.dtype, buffer=shm.buf)
            shm_view[:] = raster_arr
            use_shm = True
        except Exception:
            # Fallback: np.memmap on disk
            try:
                fd, memmap_path = tempfile.mkstemp(suffix=".dat", prefix="ganabosques_risk_")
                os.close(fd)
                mm = np.memmap(memmap_path, dtype=raster_arr.dtype, mode="w+", shape=raster_arr.shape)
                mm[:] = raster_arr[:]
                mm.flush()
                use_memmap = True
            except Exception:
                # Final fallback: streaming mode (reopen file per window)
                streaming = True

        init_kwargs = dict(
            shape=raster_arr.shape,
            dtype=str(raster_arr.dtype),
            transform=transform,
            width=width,
            height=height,
            pixel_area_ha=pixel_area_ha,
            defo_value=int(deforestation_value),
            shm_name=(shm.name if use_shm else None),
            memmap_path=(memmap_path if use_memmap else None),
            protected_union_wkb=prot_union_wkb,
            farming_union_wkb=farm_union_wkb,
            raster_path=(deforestation if streaming else None),  # <-- clave
        )

        try:
            with ProcessPoolExecutor(
                max_workers=int(n_workers),
                initializer=_init_worker,
                initargs=tuple(init_kwargs.values()),
            ) as ex:
                futures = []
                for _, row in plots.iterrows():
                    futures.append(ex.submit(_process_one, str(row[id_column]), row.geometry))
                for f in tqdm(futures, total=len(futures), desc="Computing direct alerts (parallel)"):
                    results.append(f.result())
        finally:
            if use_shm and shm is not None:
                try:
                    shm.close(); shm.unlink()
                except Exception:
                    pass
            if use_memmap and memmap_path is not None:
                try:
                    os.remove(memmap_path)
                except Exception:
                    pass

    # ------------------------
    # Build final DataFrame (stable schema/order)
    # ------------------------
    out = pd.DataFrame(results)
    # Ensure column order
    cols = [
        "id",
        "plot_area",
        "deforested_area",
        "deforested_proportion",
        "protected_areas_area",
        "protected_areas_proportion",
        "farming_in_area",
        "farming_in_proportion",
        "farming_out_area",
        "farming_out_proportion",
        "alert_direct",
    ]
    # Add any missing columns (if some plot had no geometry, etc.)
    for c in cols:
        if c not in out.columns:
            out[c] = 0 if c != "alert_direct" else False

    out = out[cols].reset_index(drop=True)
    # Cast types explicitly
    out["alert_direct"] = out["alert_direct"].astype(bool)
    for c in cols:
        if c.endswith("_proportion"):
            out[c] = out[c].clip(lower=0.0, upper=1.0)

    return out


# --------------------------------------------------------------------------------------
# Example usage (commented)
# --------------------------------------------------------------------------------------
# if __name__ == "__main__":
#     import geopandas as gpd
#     plots = gpd.read_file("tests/data_test/plot_intersect.shp")
#     if "id" not in plots.columns:
#         plots["id"] = [f"p{i}" for i in range(len(plots))]
#     out = alert_direct(
#         plots=plots,
#         deforestation="tests/data_test/deforestation.tif",
#         protected_areas="tests/data_test/areas.shp",
#         farming_areas="tests/data_test/areas.shp",
#         deforestation_value=2,
#         n_workers=2,
#     )
#     print(out.head())
