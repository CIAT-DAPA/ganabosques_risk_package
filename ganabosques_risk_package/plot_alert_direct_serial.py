# Filename: plot_alert_direct_serial.py
# Description:
#   Compute per-plot direct deforestation and land-use metrics by intersecting:
#     - a deforestation raster (pixel-class based)
#     - protected areas polygons
#     - farming areas polygons
#   This version is 100% serial (no multiprocessing), useful for debugging, logs
#   and environments where parallel execution is problematic.
#
# Public API:
#   - alert_direct(...)
#
# Author: Steven Sotelo (serial variant adapted by ChatGPT)
#
# Notes:
#   - All areas reported in hectares; proportions in [0, 1].
#   - CRS: everything is reprojected to the raster CRS; if a vector has no CRS,
#     it is assumed to already be in the raster CRS.

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.windows import transform as window_transform
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from shapely.ops import unary_union
from tqdm import tqdm


# --------------------------------------------------------------------------------------
# Module-level globals (used only in serial mode)
# --------------------------------------------------------------------------------------

_R_ARRAY: Optional[np.ndarray] = None
_R_SHAPE: Optional[Tuple[int, int]] = None
_R_DTYPE: Optional[np.dtype] = None

_R_TRANSFORM = None
_R_WIDTH: Optional[int] = None
_R_HEIGHT: Optional[int] = None

_R_PIXEL_AREA_HA: Optional[float] = None
_R_DEFO_VALUE: Optional[int] = None

_G_PROTECTED_UNION = None
_G_FARMING_UNION = None


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    """Return num / den, guarding for zero / invalid denominators."""
    if den is None or den <= 0:
        return 0.0
    return float(num) / float(den)


def _bounds_to_window(
    bounds,
    full_transform,
    raster_width: int,
    raster_height: int
) -> Tuple[int, int, int, int]:
    """Convert geometry bounds to a clamped raster window (no boundless)."""
    win = from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3],
        transform=full_transform,
    )

    full_win = Window(col_off=0, row_off=0, width=raster_width, height=raster_height)
    win = win.intersection(full_win)

    row_off = int(max(0, math.floor(win.row_off)))
    col_off = int(max(0, math.floor(win.col_off)))
    h = int(max(0, math.ceil(win.height)))
    w = int(max(0, math.ceil(win.width)))

    if row_off >= raster_height or col_off >= raster_width:
        return 0, 0, 0, 0

    h = min(h, raster_height - row_off)
    w = min(w, raster_width - col_off)
    return row_off, col_off, h, w


# --------------------------------------------------------------------------------------
# Core per-plot operations (serial)
# --------------------------------------------------------------------------------------

def _intersect_raster_deforestation(geom) -> float:
    """Compute deforested area (ha) for a single plot geometry."""
    global _R_ARRAY, _R_TRANSFORM, _R_WIDTH, _R_HEIGHT, _R_DEFO_VALUE, _R_PIXEL_AREA_HA

    if _R_ARRAY is None:
        return 0.0

    row_off, col_off, h, w = _bounds_to_window(
        geom.bounds,
        _R_TRANSFORM,
        _R_WIDTH,
        _R_HEIGHT,
    )
    if h == 0 or w == 0:
        return 0.0

    win = Window(col_off, row_off, w, h)
    win_tf = window_transform(win, _R_TRANSFORM)

    mask_inside = geometry_mask(
        [mapping(geom)],
        out_shape=(h, w),
        transform=win_tf,
        invert=True,
        all_touched=False,
    )

    view = _R_ARRAY[row_off: row_off + h, col_off: col_off + w]
    hits = (view == _R_DEFO_VALUE) & mask_inside
    pixels = int(np.count_nonzero(hits))
    return float(pixels) * float(_R_PIXEL_AREA_HA)


def _intersect_area_ha(geom, union_geom) -> float:
    """Return intersection area in hectares between 'geom' and a union geometry."""
    if union_geom is None or union_geom.is_empty or geom.is_empty:
        return 0.0
    return float(geom.intersection(union_geom).area) / 10000.0


def _process_one(plot_id: str, geom) -> Dict[str, float]:
    """Compute all metrics for a single plot geometry (serial)."""
    global _G_PROTECTED_UNION, _G_FARMING_UNION

    plot_area_ha = float(geom.area) / 10000.0

    defo_ha = _intersect_raster_deforestation(geom)
    prot_ha = _intersect_area_ha(geom, _G_PROTECTED_UNION)
    farm_in_ha = _intersect_area_ha(geom, _G_FARMING_UNION)
    farm_out_ha = max(plot_area_ha - farm_in_ha, 0.0)

    defo_prop = _safe_div(defo_ha, plot_area_ha)
    prot_prop = _safe_div(prot_ha, plot_area_ha)
    farm_in_prop = _safe_div(farm_in_ha, plot_area_ha)
    farm_out_prop = _safe_div(farm_out_ha, plot_area_ha)

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
# Public API (serial implementation)
# --------------------------------------------------------------------------------------

def alert_direct_serial(
    plots: gpd.GeoDataFrame,
    deforestation: str,
    protected_areas: str,
    farming_areas: str,
    deforestation_value: int = 2,
    id_column: str = "id",
) -> pd.DataFrame:
    """Compute per-plot direct deforestation and land-use metrics (serial).

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
    n_workers : int, default 1
        Kept only for compatibility. The computation is always serial.
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
    print(f"[Serial] Opening deforestation raster: {deforestation}")
    with rasterio.open(deforestation) as src:
        raster_arr = src.read(1)
        transform = src.transform
        width, height = src.width, src.height
        raster_crs = src.crs

        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
        pixel_area_ha = float(pixel_area_m2) / 10000.0

    # ------------------------
    # Load vector layers
    # ------------------------
    print(f"[Serial] Loading vector layers:")
    print(f"         Protected areas = {protected_areas}")
    print(f"         Farming areas   = {farming_areas}")

    prot = gpd.read_file(protected_areas)
    farm = gpd.read_file(farming_areas)

    # CRS handling for vectors
    if raster_crs is not None:
        if prot.crs is None:
            prot = prot.set_crs(raster_crs, inplace=False)
        if farm.crs is None:
            farm = farm.set_crs(raster_crs, inplace=False)

    if raster_crs is not None:
        if prot.crs != raster_crs:
            prot = prot.to_crs(raster_crs)
        if farm.crs != raster_crs:
            farm = farm.to_crs(raster_crs)

    print("[Serial] Building union geometries for protected/farming layers")
    prot_union = unary_union(prot.geometry) if (len(prot) > 0 and prot.geometry.notnull().any()) else None
    farm_union = unary_union(farm.geometry) if (len(farm) > 0 and farm.geometry.notnull().any()) else None

    # ------------------------
    # Prepare plots
    # ------------------------
    print("[Serial] Preparing plots (CRS + geometry cleaning)")
    plots = plots[[id_column, "geometry"]].copy()

    if raster_crs is not None and plots.crs is None:
        plots = plots.set_crs(raster_crs, inplace=False)

    if raster_crs is not None and plots.crs != raster_crs:
        plots = plots.to_crs(raster_crs)

    plots = plots[plots.geometry.notnull()].reset_index(drop=True)

    # ------------------------
    # Initialize module-level globals for serial run
    # ------------------------
    global _R_ARRAY, _R_SHAPE, _R_DTYPE
    global _R_TRANSFORM, _R_WIDTH, _R_HEIGHT, _R_PIXEL_AREA_HA, _R_DEFO_VALUE
    global _G_PROTECTED_UNION, _G_FARMING_UNION

    _R_ARRAY = raster_arr
    _R_SHAPE = raster_arr.shape
    _R_DTYPE = raster_arr.dtype
    _R_TRANSFORM = transform
    _R_WIDTH = width
    _R_HEIGHT = height
    _R_PIXEL_AREA_HA = float(pixel_area_ha)
    _R_DEFO_VALUE = int(deforestation_value)

    _G_PROTECTED_UNION = prot_union
    _G_FARMING_UNION = farm_union

    # ------------------------
    # Serial loop over plots
    # ------------------------
    print(f"[Serial] Computing direct alerts for {len(plots)} plots (single process)")
    results: List[Dict] = []
    for _, row in tqdm(
        plots.iterrows(),
        total=len(plots),
        desc="[Serial] Computing direct alerts",
    ):
        metrics = _process_one(plot_id=str(row[id_column]), geom=row.geometry)
        results.append(metrics)

    # ------------------------
    # Build final DataFrame
    # ------------------------
    print("[Serial] Building final DataFrame")
    out = pd.DataFrame(results)

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
    for c in cols:
        if c not in out.columns:
            out[c] = False if c == "alert_direct" else 0.0

    out = out[cols].reset_index(drop=True)
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
#
#     plots = gpd.read_file("tests/data_test/plot_intersect.shp")
#     if "id" not in plots.columns:
#         plots["id"] = [f"p{i}" for i in range(len(plots))]
#
#     out = alert_direct(
#         plots=plots,
#         deforestation="tests/data_test/deforestation.tif",
#         protected_areas="tests/data_test/areas.shp",
#         farming_areas="tests/data_test/areas.shp",
#         deforestation_value=2,
#         n_workers=1,  # ignored, always serial
#     )
#     print(out.head())
