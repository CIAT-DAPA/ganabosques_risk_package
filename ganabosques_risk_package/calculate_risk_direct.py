
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

def calculate_risk_direct(df_plots, raster_deforestation_path, shp_protected_areas_path, shp_farming_areas_path):
    """
    Calculate direct risk for each property according to MRV protocol.

    Parameters:
    - df_plots: GeoDataFrame with 'geometry' column representing the buffer areas
    - raster_deforestation_path: path to the deforestation raster (GeoTIFF)
    - shp_protected_areas_path: path to the protected areas shapefile
    - shp_farming_areas_path: path to the agricultural frontier shapefile

    Returns:
    - DataFrame with: hectares_deforested, proportion_deforested,
                      distance_to_deforestation, distance_to_protected_area,
                      distance_to_agro_frontier, and risk level
    """

    # Validate the column geometry in the df_
    if not isinstance(df_plots, gpd.GeoDataFrame):
        raise ValueError("df_plots must be a GeoDataFrame with a 'geometry' column")

    gdf_protegidas = gpd.read_file(shp_protected_areas_path).to_crs(df_plots.crs)
    gdf_frontera = gpd.read_file(shp_farming_areas_path).to_crs(df_plots.crs)

    results = []

    with rasterio.open(raster_deforestation_path) as src:
        df_plots_proj = df_plots.to_crs(src.crs)
        gdf_protegidas_proj = gdf_protegidas.to_crs(src.crs)
        gdf_frontera_proj = gdf_frontera.to_crs(src.crs)

        for idx, row in df_plots_proj.iterrows():
            geom = row.geometry
            area_total_ha = geom.area / 10000.0

            out_image, _ = rasterio.mask.mask(src, [geom], crop=True)
            out_image = out_image[0]
            deforested_pixels = np.sum(out_image == 1)
            pixel_area = src.res[0] * src.res[1]
            hectares_deforested = deforested_pixels * pixel_area / 10000.0
            proportion_deforested = hectares_deforested / area_total_ha if area_total_ha > 0 else 0

            deforestation_coords = np.argwhere(out_image == 1)
            min_dist_deforest = float('nan')
            if deforestation_coords.size > 0:
                min_dist_deforest = 0  # Raster overlaps; direct match assumed since geometry encloses it

            min_dist_protected = gdf_protegidas_proj.distance(geom).min()
            min_dist_agro = gdf_frontera_proj.distance(geom).min()

            risk_level = "ALTO" if hectares_deforested > 0 and min_dist_protected < 500 else "BAJO"

            results.append({
                "id": row.get("id", idx),
                "hectares_deforested": hectares_deforested,
                "proportion_deforested": proportion_deforested,
                "distance_to_deforestation": min_dist_deforest,
                "distance_to_protected_area": min_dist_protected,
                "distance_to_agro_frontier": min_dist_agro,
                "risk_direct_level": risk_level
            })

    return pd.DataFrame(results)
