# Filename: plot_alert_direct.py
# Description:
#   Compute direct alerts and enviromentals indicators for plots
#   based on deforestation, farming areas and protected areas
#
# Public API:
#   - alert_direct(plots: gpd.GeoDataFrame,deforestation: str,protected_areas: str,farming_areas: str,deforestation_value: float = 2,
#                   n_workers: int = 2, id_column: str = "id") -> pd.DataFrame
#
# Author: Steven Sotelo
# Notes:
#   - Uses pandas/numpy/tqdm and optional multiprocessing for assigning results back to the alert_direct table.
#   - Progress is displayed with tqdm over chunked processing of plot IDs.
#import warnings
#warnings.filterwarnings("ignore")

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from typing import Tuple, Optional
from shapely import wkb

import geopandas as gpd
import rasterio
import rasterio.mask
import rasterio.features
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Global variables used by worker processes
# ----------------------------
_G_RASTER_SHM_NAME: Optional[str] = None
_G_RASTER_SHAPE: Optional[Tuple[int, int]] = None
_G_RASTER_DTYPE: Optional[str] = None
_G_RASTER_NODATA: Optional[float] = None
_G_RASTER_TRANSFORM_AFFINE_PARAMS: Optional[Tuple[float, float, float, float, float, float]] = None
_G_PIXEL_AREA_HA: Optional[float] = None
_G_DEFO_VALUE: Optional[float] = None

_G_PROTECTED_GDF: Optional[gpd.GeoDataFrame] = None
_G_FARMING_GDF: Optional[gpd.GeoDataFrame] = None


def _init_worker(
    shm_name: str,
    raster_shape: Tuple[int, int],
    raster_dtype: str,
    raster_nodata: Optional[float],
    affine_params: Tuple[float, float, float, float, float, float],
    pixel_area_ha: float,
    defo_value: float,
    protected_gdf: gpd.GeoDataFrame,
    farming_gdf: gpd.GeoDataFrame,
):
    """Initialize worker process with shared raster and vector layer references."""
    global _G_RASTER_SHM_NAME, _G_RASTER_SHAPE, _G_RASTER_DTYPE, _G_RASTER_NODATA
    global _G_RASTER_TRANSFORM_AFFINE_PARAMS, _G_PIXEL_AREA_HA, _G_DEFO_VALUE
    global _G_PROTECTED_GDF, _G_FARMING_GDF

    _G_RASTER_SHM_NAME = shm_name
    _G_RASTER_SHAPE = raster_shape
    _G_RASTER_DTYPE = raster_dtype
    _G_RASTER_NODATA = raster_nodata
    _G_RASTER_TRANSFORM_AFFINE_PARAMS = affine_params
    _G_PIXEL_AREA_HA = pixel_area_ha
    _G_DEFO_VALUE = defo_value

    _G_PROTECTED_GDF = protected_gdf
    if _G_PROTECTED_GDF is not None and len(_G_PROTECTED_GDF) > 0:
        _G_PROTECTED_GDF.sindex

    _G_FARMING_GDF = farming_gdf
    if _G_FARMING_GDF is not None and len(_G_FARMING_GDF) > 0:
        _G_FARMING_GDF.sindex


def _get_raster_array_view() -> np.ndarray:
    """Return a shared-memory view of the raster array."""
    assert _G_RASTER_SHM_NAME and _G_RASTER_SHAPE and _G_RASTER_DTYPE
    shm = shared_memory.SharedMemory(name=_G_RASTER_SHM_NAME)
    return np.ndarray(_G_RASTER_SHAPE, dtype=_G_RASTER_DTYPE, buffer=shm.buf)


def _bounds_to_window(bounds, full_transform, raster_width, raster_height):
    """Convert geometry bounds to a raster window clipped to raster limits."""
    win = from_bounds(*bounds, transform=full_transform, width=raster_width, height=raster_height, boundless=True)
    row_off = max(0, int(np.floor(win.row_off)))
    col_off = max(0, int(np.floor(win.col_off)))
    height = min(raster_height - row_off, int(np.ceil(win.height)))
    width = min(raster_width - col_off, int(np.ceil(win.width)))
    height = max(0, height)
    width = max(0, width)
    return (row_off, col_off, height, width)


def _intersect_raster_deforestation(geom, raster_arr: np.ndarray) -> float:
    """Compute deforested area (ha) inside a polygon using the in-memory raster."""
    from affine import Affine
    transform = Affine(*_G_RASTER_TRANSFORM_AFFINE_PARAMS)
    height, width = raster_arr.shape

    # Determine raster window around the polygon
    row_off, col_off, h, w = _bounds_to_window(geom.bounds, transform, width, height)
    if h == 0 or w == 0:
        return 0.0

    r_slice = raster_arr[row_off:row_off + h, col_off:col_off + w]
    win = rasterio.windows.Window(col_off, row_off, w, h)
    win_transform = window_transform(win, transform)

    mask = rasterio.features.geometry_mask(
        [geom.__geo_interface__],
        out_shape=(h, w),
        transform=win_transform,
        invert=True
    )

    # Handle NoData pixels
    if _G_RASTER_NODATA is not None:
        valid = r_slice != _G_RASTER_NODATA
    else:
        valid = np.ones_like(r_slice, dtype=bool)

    cond = (r_slice == _G_DEFO_VALUE) & valid & mask
    count = int(np.count_nonzero(cond))
    return count * _G_PIXEL_AREA_HA


def _intersect_area_layer(geom, layer_gdf: Optional[gpd.GeoDataFrame]) -> float:
    """Compute intersection area (ha) between polygon and vector layer."""
    if layer_gdf is None or len(layer_gdf) == 0:
        return 0.0

    idx = list(layer_gdf.sindex.intersection(geom.bounds))
    if not idx:
        return 0.0

    candidates = layer_gdf.iloc[idx]
    inters = candidates.intersection(geom)
    area = float(np.sum([g.area for g in inters if g is not None and not g.is_empty]))
    return area / 10000.0  # convert m² → ha


def _process_one(record: Tuple):
    """Process one plot polygon and compute all metrics."""
    plot_id, geom_wkb = record
    geom = wkb.loads(geom_wkb)

    raster_arr = _get_raster_array_view()
    plot_area_ha = geom.area / 10000.0

    # --- Deforestation area ---
    defo_area_ha = _intersect_raster_deforestation(geom, raster_arr)
    defo_prop = (defo_area_ha / plot_area_ha) if plot_area_ha > 0 else 0.0

    # New field: alert_direct is True if deforested_area > 0
    alert_direct = defo_area_ha > 0

    # --- Protected areas intersection ---
    protected_area_ha = _intersect_area_layer(geom, _G_PROTECTED_GDF)
    protected_prop = (protected_area_ha / plot_area_ha) if plot_area_ha > 0 else 0.0

    # --- Farming areas intersection ---
    farming_in_ha = _intersect_area_layer(geom, _G_FARMING_GDF)
    farming_in_ha = max(0.0, min(farming_in_ha, plot_area_ha))
    farming_out_ha = max(0.0, plot_area_ha - farming_in_ha)
    farming_in_prop = (farming_in_ha / plot_area_ha) if plot_area_ha > 0 else 0.0
    farming_out_prop = (farming_out_ha / plot_area_ha) if plot_area_ha > 0 else 0.0

    # Return all computed values as a dictionary
    return {
        "id": plot_id,
        "plot_area": plot_area_ha,
        "deforested_area": defo_area_ha,
        "deforested_proportion": defo_prop,
        "protected_areas_area": protected_area_ha,
        "protected_areas_proportion": protected_prop,
        "farming_in_area": farming_in_ha,
        "farming_in_proportion": farming_in_prop,
        "farming_out_area": farming_out_ha,
        "farming_out_proportion": farming_out_prop,
        "alert_direct": alert_direct,
    }


def alert_direct(
    plots: gpd.GeoDataFrame,
    deforestation: str,
    protected_areas: str,
    farming_areas: str,
    deforestation_value: float = 2,
    n_workers: int = 2,
    id_column: str = "id",
) -> pd.DataFrame:
    """Compute overlay metrics for each plot including deforestation alerts."""
    if id_column not in plots.columns:
        raise ValueError(f"The 'plots' GeoDataFrame must contain column '{id_column}'.")

    # --- Load raster once ---
    with rasterio.open(deforestation) as src:
        raster_crs = src.crs
        transform = src.transform
        nodata = src.nodata
        raster_arr = src.read(1)
        pixel_area_ha = abs(transform.a * transform.e) / 10000.0

    # --- Load vector layers and align CRS ---
    protected_gdf = gpd.read_file(protected_areas) if protected_areas else gpd.GeoDataFrame(geometry=[])
    farming_gdf = gpd.read_file(farming_areas) if farming_areas else gpd.GeoDataFrame(geometry=[])

    plots_crs = plots.to_crs(raster_crs)
    protected_crs = protected_gdf.to_crs(raster_crs) if len(protected_gdf) else protected_gdf
    farming_crs = farming_gdf.to_crs(raster_crs) if len(farming_gdf) else farming_gdf

    # --- Shared memory for raster ---
    shm = shared_memory.SharedMemory(create=True, size=raster_arr.nbytes)
    shm_arr = np.ndarray(raster_arr.shape, dtype=raster_arr.dtype, buffer=shm.buf)
    shm_arr[:] = raster_arr

    records = list(zip(plots_crs[id_column].tolist(), plots_crs.geometry.apply(lambda g: g.wkb).tolist()))
    affine_params = (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)

    try:
        with ProcessPoolExecutor(max_workers=int(max(1, n_workers))) as ex:
            ex._initializer = _init_worker
            ex._initargs = (
                shm.name,
                raster_arr.shape,
                str(raster_arr.dtype),
                nodata,
                affine_params,
                pixel_area_ha,
                float(deforestation_value),
                protected_crs,
                farming_crs,
            )

            results = []
            for out in tqdm(ex.map(_process_one, records), total=len(records), desc="Processing plots"):
                results.append(out)

    finally:
        shm.close()
        shm.unlink()

    # --- Final DataFrame ---
    df = pd.DataFrame(results)
    df = df[
        [
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
    ].sort_values(by="id")

    return df


# ----------------------------
# Example usage (commented out)
# ----------------------------
# plots_gdf = gpd.read_file("plots.shp")
# df_out = compute_plot_overlays(
#     plots=plots_gdf,
#     deforestation="deforestation.tif",
#     protected_areas="protected.shp",
#     farming_areas="farming.shp",
#     deforestation_value=2,
#     n_workers=4,
# )
# print(df_out.head())
