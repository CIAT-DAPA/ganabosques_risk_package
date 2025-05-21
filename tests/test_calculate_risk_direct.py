import unittest
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
from ganabosques_risk_package.calculate_risk_direct import calculate_risk_direct

class TestCalculateRiskDirect(unittest.TestCase):
    def setUp(self):
        # Crear un GeoDataFrame ficticio con un polígono cuadrado
        self.polygon = Polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0)])
        self.df_plots = gpd.GeoDataFrame({
            "id": [1],
            "geometry": [self.polygon]
        }, crs="EPSG:3857")

        # Archivos falsos que no serán cargados (solo para prueba de errores)
        self.raster_path = "fake_raster.tif"
        self.shp_protegidas = "fake_protected.shp"
        self.shp_frontera = "fake_frontera.shp"

    def test_error_with_non_geodataframe(self):
        # Comentario: prueba que lanza error si no se pasa un GeoDataFrame
        with self.assertRaises(ValueError):
            calculate_risk_direct(pd.DataFrame(), self.raster_path, self.shp_protegidas, self.shp_frontera)

    def test_placeholder_geometry(self):
        # Comentario: no se prueba la lógica espacial aquí, solo aseguramos estructura general
        self.assertIn("geometry", self.df_plots.columns)
        self.assertEqual(len(self.df_plots), 1)
        self.assertTrue(self.df_plots.geometry.iloc[0].is_valid)

# Ejecutar solo si este archivo se ejecuta directamente
if __name__ == "__main__":
    unittest.main()