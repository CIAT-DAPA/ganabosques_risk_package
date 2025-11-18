# Filename: plot_alert_direct_parallel.py
# Description:
#   Parallel implementation of direct alerts per plot:
#     - Deforestation (raster) → zonal statistics via rasterstats.zonal_stats
#     - Protected areas (vector) → union + intersect per plot
#     - Farming areas (vector) → union + intersect per plot
#
# Public API:
#   - alert_direct_parallel(...)
#
# Notes:
#   - All areas are reported in hectares.
#   - Proportions are in [0, 1].
#   - CRS: plots / protected / farming are all reprojected to the raster CRS.
#   - Parallelization:
#       * zonal_stats is run once (serial) on the full GeoDataFrame.
#       * per-plot aggregation (protected/farming intersections + metrics) is
#         split across worker processes using simple Python lists + WKB geometries
#         to avoid heavy GeoDataFrame pickling.
#
# Author: Steven Sotelo (adapted, parallelized by ChatGPT)

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.ops import unary_union
from shapely.wkb import loads as wkb_loads
from tqdm import tqdm
from rasterstats import zonal_stats
from concurrent.futures import ProcessPoolExecutor, as_completed


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def _safe_div(num: float, den: float) -> float:
    """Return num / den guarding against division by zero or invalid denominators."""
    if den is None or den <= 0:
        return 0.0
    return float(num) / float(den)


def _intersect_area_ha(geom, union_geom) -> float:
    """Return intersection area in hectares between `geom` and a union geometry."""
    if union_geom is None or union_geom.is_empty or geom is None or geom.is_empty:
        return 0.0
    return float(geom.intersection(union_geom).area) / 10000.0


def _chunk_indices(n_items: int, n_workers: int) -> List[Tuple[int, int]]:
    """
    Generate (start, end) index ranges that partition [0, n_items) into
    approximately equal contiguous chunks for parallel processing.
    """
    n_workers = max(1, int(n_workers))
    if n_items == 0:
        return []
    if n_workers > n_items:
        n_workers = n_items
    chunk_size = math.ceil(n_items / n_workers)
    chunks: List[Tuple[int, int]] = []
    for start in range(0, n_items, chunk_size):
        end = min(start + chunk_size, n_items)
        chunks.append((start, end))
    return chunks


def _process_chunk(
    ids_chunk: List[str],
    areas_chunk: List[float],
    geoms_wkb_chunk: List[bytes],
    zs_chunk: List[Dict],
    prot_union,
    farm_union,
    pixel_area_ha: float,
    deforestation_value: int,
) -> List[Dict]:
    """
    Worker function: compute all per-plot metrics for a subset of plots.

    Args:
        ids_chunk: list of plot IDs (strings).
        areas_chunk: list of plot areas in hectares.
        geoms_wkb_chunk: list of geometries serialized as WKB bytes.
        zs_chunk: list of zonal_stats dicts, aligned with ids/geoms/areas.
        prot_union: unary_union of protected_areas geometries.
        farm_union: unary_union of farming_areas geometries.
        pixel_area_ha: area of one raster pixel in hectares.
        deforestation_value: pixel value representing deforestation.

    Returns:
        List of dicts, each with metrics for one plot.
    """
    out: List[Dict] = []

    for plot_id, plot_area_ha, geom_wkb, zcat in zip(
        ids_chunk, areas_chunk, geoms_wkb_chunk, zs_chunk
    ):
        geom = wkb_loads(geom_wkb)

        # Deforestation: number of pixels == deforestation_value
        defo_pixels = 0
        if isinstance(zcat, dict):
            defo_pixels = int(
                zcat.get(int(deforestation_value), 0)
                or zcat.get(str(deforestation_value), 0)
                or 0
            )
        defo_ha = float(defo_pixels) * float(pixel_area_ha)
        defo_prop = _safe_div(defo_ha, plot_area_ha)

        # Protected / farming areas: vector–vector intersections (using unions)
        prot_ha = _intersect_area_ha(geom, prot_union)
        prot_prop = _safe_div(prot_ha, plot_area_ha)

        farm_in_ha = _intersect_area_ha(geom, farm_union)
        farm_in_prop = _safe_div(farm_in_ha, plot_area_ha)

        farm_out_ha = max(plot_area_ha - farm_in_ha, 0.0)
        farm_out_prop = _safe_div(farm_out_ha, plot_area_ha)

        alert = bool(defo_ha > 0.0)

        out.append(
            {
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
        )

    return out


# --------------------------------------------------------------------------------------
# Public API (parallel, zonal_stats for deforestation)
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
    """Compute per-plot direct deforestation + land-use metrics (parallel, zonal_stats).

    Parallelization strategy (robust, sin shared memory):

      1. Se abre el raster una sola vez para obtener CRS y pixel_area_ha.
      2. Se cargan protected_areas y farming_areas, se re-proyectan al CRS del raster
         y se construyen sus unary_union.
      3. Se re-proyectan los plots al CRS del raster y se calcula plot_area (ha).
      4. Se ejecuta rasterstats.zonal_stats **una sola vez** (serial) sobre todos
         los plots (polígonos).
      5. Se crean listas simples:
           - ids: List[str]
           - areas: List[float]
           - geoms_wkb: List[bytes]
           - zs: List[dict]
         y se particionan en chunks.
      6. Cada worker recibe sólo listas Python + las uniones (prot_union, farm_union)
         y calcula las métricas de su subconjunto.

    Si n_workers <= 1, la función cae en modo completamente serial.

    Parameters
    ----------
    plots : geopandas.GeoDataFrame
        Plot polygons. Must contain an ID column (default 'id') and a geometry column.
        CRS can be anything; it will be reprojected to the raster CRS.
    deforestation : str
        Path to a raster file (e.g. GeoTIFF) containing a deforestation class code.
    protected_areas : str
        Path to a polygon vector dataset (SHP/GeoJSON) for protected areas.
    farming_areas : str
        Path to a polygon vector dataset (SHP/GeoJSON) for farming areas.
    deforestation_value : int, default 2
        Pixel value that encodes “deforestation” in the raster.
    n_workers : int, default 2
        Number of worker processes for parallel per-plot aggregation.
        If n_workers <= 1, computation falls back to pure serial.
    id_column : str, default "id"
        Name of the plot ID column in `plots`.

    Returns
    -------
    pandas.DataFrame
        One row per plot with:
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

    # --------------------------------------------------------------------------
    # 1. Open raster (to get CRS + pixel area)
    # --------------------------------------------------------------------------
    print(f"[Parallel/zonal_stats] Opening deforestation raster: {deforestation}")
    with rasterio.open(deforestation) as src:
        raster_crs = src.crs
        transform = src.transform
        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
        pixel_area_ha = float(pixel_area_m2) / 10000.0

    # --------------------------------------------------------------------------
    # 2. Load vector layers and project to raster CRS
    # --------------------------------------------------------------------------
    print("[Parallel/zonal_stats] Loading vector layers:")
    print(f"  - Protected areas: {protected_areas}")
    print(f"  - Farming areas  : {farming_areas}")

    prot = gpd.read_file(protected_areas)
    farm = gpd.read_file(farming_areas)

    if raster_crs is not None:
        if prot.crs is None:
            prot = prot.set_crs(raster_crs, inplace=False)
        if farm.crs is None:
            farm = farm.set_crs(raster_crs, inplace=False)

        if prot.crs != raster_crs:
            prot = prot.to_crs(raster_crs)
        if farm.crs != raster_crs:
            farm = farm.to_crs(raster_crs)

    print("[Parallel/zonal_stats] Building union geometries (protected, farming)")
    prot_union = unary_union(prot.geometry) if (len(prot) > 0 and prot.geometry.notnull().any()) else None
    farm_union = unary_union(farm.geometry) if (len(farm) > 0 and farm.geometry.notnull().any()) else None

    # --------------------------------------------------------------------------
    # 3. Prepare plots (reproject to raster CRS) and compute basic areas
    # --------------------------------------------------------------------------
    print("[Parallel/zonal_stats] Preparing plots (reproject to raster CRS)")
    plots = plots[[id_column, "geometry"]].copy()

    if raster_crs is not None:
        if plots.crs is None:
            plots = plots.set_crs(raster_crs, inplace=False)
        elif plots.crs != raster_crs:
            plots = plots.to_crs(raster_crs)

    plots = plots[plots.geometry.notnull()].reset_index(drop=True)

    if len(plots) == 0:
        print("[Parallel/zonal_stats] No plots to process. Returning empty result.")
        empty = pd.DataFrame(
            columns=[
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
        )
        return empty

    print("[Parallel/zonal_stats] Computing plot areas (ha)")
    plots["plot_area"] = plots.geometry.area / 10000.0

    # --------------------------------------------------------------------------
    # 4. Zonal statistics for deforestation using rasterstats (serial)
    # --------------------------------------------------------------------------
    print("[Parallel/zonal_stats] Running zonal_stats for deforestation (categorical)...")
    zs = zonal_stats(
        plots,
        deforestation,
        categorical=True,
        all_touched=False,
        nodata=None,
        affine=None,  # infer from file
    )
    if len(zs) != len(plots):
        raise RuntimeError("zonal_stats result length does not match number of plots.")

    # --------------------------------------------------------------------------
    # 5. Prepare simple lists for multiprocessing
    # --------------------------------------------------------------------------
    ids = plots[id_column].astype(str).tolist()
    areas = plots["plot_area"].astype(float).tolist()
    geoms_wkb = [geom.wkb for geom in plots.geometry]
    N = len(ids)

    n_workers = int(n_workers)
    if n_workers <= 1 or N == 0:
        print(f"[Parallel/zonal_stats] Falling back to serial mode for {N} plots")
        results = _process_chunk(
            ids_chunk=ids,
            areas_chunk=areas,
            geoms_wkb_chunk=geoms_wkb,
            zs_chunk=zs,
            prot_union=prot_union,
            farm_union=farm_union,
            pixel_area_ha=pixel_area_ha,
            deforestation_value=deforestation_value,
        )
    else:
        print(f"[Parallel/zonal_stats] Computing metrics in parallel with {n_workers} workers for {N} plots")
        chunks = _chunk_indices(N, n_workers)
        results: List[Dict] = []

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = []
            for start, end in chunks:
                ids_chunk = ids[start:end]
                areas_chunk = areas[start:end]
                geoms_wkb_chunk = geoms_wkb[start:end]
                zs_chunk = zs[start:end]

                fut = ex.submit(
                    _process_chunk,
                    ids_chunk,
                    areas_chunk,
                    geoms_wkb_chunk,
                    zs_chunk,
                    prot_union,
                    farm_union,
                    pixel_area_ha,
                    deforestation_value,
                )
                futures.append(fut)

            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[Parallel/zonal_stats] Aggregating chunks",
            ):
                results.extend(fut.result())

    # --------------------------------------------------------------------------
    # 6. Build final DataFrame
    # --------------------------------------------------------------------------
    print("[Parallel/zonal_stats] Building final DataFrame")
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
#     out = alert_direct_parallel(
#         plots=plots,
#         deforestation="tests/data_test/deforestation.tif",
#         protected_areas="tests/data_test/areas.shp",
#         farming_areas="tests/data_test/areas.shp",
#         deforestation_value=2,
#         n_workers=4,
#     )
#     print(out.head())
