"""
Module: calculate_risk_direct
Description: Implements direct risk assessment based on deforestation, proximity to protected areas,
and agricultural frontier using the MRV protocol. Assumes input geometries include pre-defined buffers.
"""

import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
import pandas as pd
from tqdm import tqdm

from enum import Enum

class RiskLevel(Enum):
    """
    Enum: RiskLevel
    Description:
        Represents the categorical levels of direct deforestation risk
        according to the MRV protocol used in the Ganabosques system.

    Values:
        - LOW: Low risk of deforestation.
        - MEDIUM: Medium risk of deforestation.
        - HIGH: High risk of deforestation.
        - NO_RISK: No risk of deforestation.
    """
    LOW = "BAJO"
    MEDIUM = "MEDIO"
    HIGH = "ALTO"
    NO_RISK = "SIN RIESGO"

def count_deforestation_in_ring(src, geometry, deforested_value):
    """
    Count number of deforestation pixels (value == deforested_value) in a given geometry.
    """
    out_image, _ = rasterio.mask.mask(src, [geometry], crop=True)
    out_image = out_image[0]
    return np.count_nonzero(out_image == deforested_value)

def calculate_risk_direct(df_plots, raster_deforestation_path, shp_protected_areas_path, shp_farming_areas_path, deforested_value = 2):
    """
    Calculate direct risk for each property according to MRV protocol.

    Parameters:
    - df_plots: GeoDataFrame with 'geometry' column representing the buffer areas. It also should have a column 'id' which identifies the record
    - raster_deforestation_path: path to the deforestation raster (GeoTIFF)
    - shp_protected_areas_path: path to the protected areas shapefile
    - shp_farming_areas_path: path to the agricultural frontier shapefile
    - deforested_value: It is the value in the raster that is considered deforestation. It is by default 2

    Returns:
    - DataFrame with: hectares_deforested, proportion_deforested,
                      distance_to_deforestation, distance_to_protected_area,
                      distance_to_agro_frontier, and risk level
    """

    # Validate the input is a GeoDataFrame
    if not isinstance(df_plots, gpd.GeoDataFrame):
        raise ValueError("df_plots must be a GeoDataFrame with a 'geometry' column")

    # Load and reproject auxiliary shapefiles to match the CRS of the plots
    gdf_protected = gpd.read_file(shp_protected_areas_path).to_crs(df_plots.crs)
    gdf_frontier = gpd.read_file(shp_farming_areas_path).to_crs(df_plots.crs)

    results = []

    # Open the deforestation raster
    with rasterio.open(raster_deforestation_path) as src:
        df_plots_proj = df_plots.to_crs(src.crs)
        gdf_protected_proj = gdf_protected.to_crs(src.crs)
        gdf_frontier_proj = gdf_frontier.to_crs(src.crs)

        # Iterate over each plot polygon using tqdm for progress indication
        for idx, row in tqdm(df_plots_proj.iterrows(), total=len(df_plots_proj), desc="Calculating risk for " + str(len(df_plots_proj)) + " plots"):
            geom = row.geometry
            # Convert m² to hectares
            area_total_ha = geom.area / 10000.0

            # Define distance buffers for deforestation
            buffer_500 = geom.buffer(500)
            buffer_2000 = geom.buffer(2000)
            buffer_5000 = geom.buffer(5000)

            # Create distance zones
            ring_0_500 = buffer_500
            ring_500_2000 = buffer_2000.difference(buffer_500)
            ring_2000_5000 = buffer_5000.difference(buffer_2000)

            pixel_area = src.res[0] * src.res[1]

            # Calculate deforestation per ring
            deforestation_0_500m = count_deforestation_in_ring(src, ring_0_500, deforested_value) * pixel_area / 10000.0
            deforestation_500_2000m = count_deforestation_in_ring(src, ring_500_2000, deforested_value) * pixel_area / 10000.0
            deforestation_2000_5000m = count_deforestation_in_ring(src, ring_2000_5000, deforested_value) * pixel_area / 10000.0

            # Clip the raster to the current geometry
            out_image, _ = rasterio.mask.mask(src, [geom], crop=True)
            out_image = out_image[0]

            # Count deforested pixels (value == deforested_value)
            deforested_pixels = np.sum(out_image == deforested_value)
            # Calculate direct deforestation
            hectares_deforested = deforested_pixels * pixel_area / 10000.0
            proportion_deforested = hectares_deforested / area_total_ha if area_total_ha > 0 else 0

            # Compute distance to deforestation (if overlap exists, assume zero)
            deforestation_coords = np.argwhere(out_image == 1)
            min_dist_deforest = float('nan')
            if deforestation_coords.size > 0:
                min_dist_deforest = 0

            # Distance-based conditions
            dist_protected = gdf_protected_proj.distance(geom).min()
            dist_frontier = gdf_frontier_proj.distance(geom).min()

            # Distance to protected areas
            protected_within_500 = dist_protected < 500
            protected_500_2000 = 500 <= dist_protected < 2000
            protected_over_2000 = dist_protected >= 2000

            in_agro_frontier = gdf_frontier_proj.intersects(geom).any()

            # Risk classification logic
            risk_level = RiskLevel.NO_RISK  # Default fallback
            if (hectares_deforested > 0 or deforestation_0_500m > 0) and protected_within_500:
                risk_level = RiskLevel.HIGH
            elif deforestation_500_2000m > 0 and protected_500_2000:
                risk_level = RiskLevel.MEDIUM
            elif deforestation_2000_5000m > 0 and protected_over_2000 and in_agro_frontier:
                risk_level = RiskLevel.LOW


            # Store result for this plot
            results.append({
                "id": row.get("id", idx),
                "deforested_hectares": hectares_deforested,
                "deforested_proportion": proportion_deforested,
                "deforested_distance_to": min_dist_deforest,
                "protected_area_distance_to": dist_protected,
                "agro_frontier_distance_to": dist_frontier,
                "risk_direct_level": risk_level.name,
                "risk_direct_level_label": risk_level.value,
                "risk_context": {
                    "deforestation_0_500m": deforestation_0_500m,
                    "deforestation_500_2000m": deforestation_500_2000m,
                    "deforestation_2000_5000m": deforestation_2000_5000m,
                    "protected_within_500": protected_within_500,
                    "protected_500_2000": protected_500_2000,
                    "protected_over_2000": protected_over_2000
                }
            })

    return pd.DataFrame(results)
