
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

def calculate_risk_direct(df_predios, raster_deforestacion_path, shp_areas_protegidas_path, shp_frontera_agro_path):
    """
    Calculate direct risk for each property according to MRV protocol.

    Parameters:
    - df_predios: GeoDataFrame with 'geometry' column representing the buffer areas
    - raster_deforestacion_path: path to the deforestation raster (GeoTIFF)
    - shp_areas_protegidas_path: path to the protected areas shapefile
    - shp_frontera_agro_path: path to the agricultural frontier shapefile

    Returns:
    - DataFrame with: hectares_deforested, proportion_deforested,
                      distance_to_deforestation, distance_to_protected_area,
                      distance_to_agro_frontier, and risk level
    """

    if not isinstance(df_predios, gpd.GeoDataFrame):
        raise ValueError("df_predios must be a GeoDataFrame with a 'geometry' column")

    gdf_protegidas = gpd.read_file(shp_areas_protegidas_path).to_crs(df_predios.crs)
    gdf_frontera = gpd.read_file(shp_frontera_agro_path).to_crs(df_predios.crs)

    results = []

    with rasterio.open(raster_deforestacion_path) as src:
        df_predios_proj = df_predios.to_crs(src.crs)
        gdf_protegidas_proj = gdf_protegidas.to_crs(src.crs)
        gdf_frontera_proj = gdf_frontera.to_crs(src.crs)

        for idx, row in df_predios_proj.iterrows():
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
