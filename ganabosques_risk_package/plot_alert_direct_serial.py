# Filename: plot_alert_direct_serial.py
# Description:
#   Serial implementation of direct alerts per plot:
#     - Deforestation (raster) → zonal statistics via rasterstats.zonal_stats
#     - Protected areas (vector) → union + intersect per plot
#     - Farming areas (vector) → union + intersect per plot
#
# Public API:
#   - alert_direct_serial(...)
#
# Notes:
#   - All areas are reported in hectares.
#   - Proportions are in [0, 1].
#   - CRS: plots / protected / farming are all reprojected to the raster CRS.
#
# Author: Steven Sotelo (adapted to zonal_stats variant)

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.ops import unary_union
from shapely.geometry import mapping
from tqdm import tqdm
from rasterstats import zonal_stats


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


# --------------------------------------------------------------------------------------
# Public API (serial, zonal_stats for deforestation)
# --------------------------------------------------------------------------------------

def alert_direct_serial(
    plots: gpd.GeoDataFrame,
    deforestation: str,
    protected_areas: str,
    farming_areas: str,
    deforestation_value: int = 2,
    n_workers: int = 1,  # kept for compatibility; ignored (always serial)
    id_column: str = "id",
) -> pd.DataFrame:
    """Compute per-plot direct deforestation + land-use metrics (serial, zonal_stats).

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
    n_workers : int, default 1
        Kept only for API compatibility. Computation is fully serial.
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

    # ------------------------------------------------------------------------------
    # 1. Open raster (to get CRS + pixel area)
    # ------------------------------------------------------------------------------
    print(f"[Serial/zonal_stats] Opening deforestation raster: {deforestation}")
    with rasterio.open(deforestation) as src:
        raster_crs = src.crs
        transform = src.transform
        # Pixel area in m² (affine determinant)
        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)
        pixel_area_ha = float(pixel_area_m2) / 10000.0

    # ------------------------------------------------------------------------------
    # 2. Load vector layers and project to raster CRS
    # ------------------------------------------------------------------------------
    print("[Serial/zonal_stats] Loading vector layers:")
    print(f"  - Protected areas: {protected_areas}")
    print(f"  - Farming areas  : {farming_areas}")

    prot = gpd.read_file(protected_areas)
    farm = gpd.read_file(farming_areas)

    # Ensure CRS for vectors
    if raster_crs is not None:
        if prot.crs is None:
            prot = prot.set_crs(raster_crs, inplace=False)
        if farm.crs is None:
            farm = farm.set_crs(raster_crs, inplace=False)

        if prot.crs != raster_crs:
            prot = prot.to_crs(raster_crs)
        if farm.crs != raster_crs:
            farm = farm.to_crs(raster_crs)

    # Build union geometries for vector–vector intersect
    print("[Serial/zonal_stats] Building union geometries (protected, farming)")
    prot_union = unary_union(prot.geometry) if (len(prot) > 0 and prot.geometry.notnull().any()) else None
    farm_union = unary_union(farm.geometry) if (len(farm) > 0 and farm.geometry.notnull().any()) else None

    # ------------------------------------------------------------------------------
    # 3. Prepare plots (reproject to raster CRS) and compute basic areas
    # ------------------------------------------------------------------------------
    print("[Serial/zonal_stats] Preparing plots (reproject to raster CRS)")
    plots = plots[[id_column, "geometry"]].copy()

    if raster_crs is not None:
        if plots.crs is None:
            plots = plots.set_crs(raster_crs, inplace=False)
        elif plots.crs != raster_crs:
            plots = plots.to_crs(raster_crs)

    plots = plots[plots.geometry.notnull()].reset_index(drop=True)

    # Precompute plot areas in ha (used for all metrics)
    print("[Serial/zonal_stats] Computing plot areas (ha)")
    plots["plot_area"] = plots.geometry.area / 10000.0

    # ------------------------------------------------------------------------------
    # 4. Zonal statistics for deforestation using rasterstats
    # ------------------------------------------------------------------------------
    print("[Serial/zonal_stats] Running zonal_stats for deforestation (categorical)...")
    # zonal_stats opens the raster internally; we just pass the path.
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

    # ------------------------------------------------------------------------------
    # 5. Per-plot loop to compute all metrics (serial)
    # ------------------------------------------------------------------------------
    print(f"[Serial/zonal_stats] Computing direct alerts for {len(plots)} plots (serial)")
    results: List[Dict] = []

    for (idx, row), zcat in tqdm(zip(plots.iterrows(), zs), total=len(plots), desc="[Serial/zonal_stats] Aggregating metrics",):
        geom = row.geometry
        plot_id = str(row[id_column])
        plot_area_ha = float(row["plot_area"])

        # Deforestation: number of pixels == deforestation_value
        # zcat is a dict { value: count, ... }
        defo_pixels = 0
        if isinstance(zcat, dict):
            # keys may be ints or strings depending on rasterstats version
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

        results.append(
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

    # ------------------------------------------------------------------------------
    # 6. Build final DataFrame
    # ------------------------------------------------------------------------------
    print("[Serial/zonal_stats] Building final DataFrame")
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

    # Clip proportions to [0, 1] just in case of numerical noise
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
#     out = alert_direct_serial(
#         plots=plots,
#         deforestation="tests/data_test/deforestation.tif",
#         protected_areas="tests/data_test/areas.shp",
#         farming_areas="tests/data_test/areas.shp",
#         deforestation_value=2,
#         n_workers=1,  # ignored (always serial)
#     )
#     print(out.head())
