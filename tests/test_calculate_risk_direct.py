import unittest
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from rasterio.io import MemoryFile
import rasterio
from rasterio.transform import from_origin
from tempfile import TemporaryDirectory
import fiona
from ganabosques_risk_package.calculate_risk_direct import calculate_risk_direct

class TestCalculateRiskDirect(unittest.TestCase):

    def setUp(self):
        # Dummy square polygon
        self.polygon = Polygon([(0, 0), (0, 1000), (1000, 1000), (1000, 0)])
        self.df_plots = gpd.GeoDataFrame({
            "id": [1],
            "geometry": [self.polygon]
        }, crs="EPSG:3857")

        # Dummy input paths
        self.raster_path = "fake_raster.tif"
        self.shp_protected = "fake_protected.shp"
        self.shp_frontier = "fake_frontier.shp"

    def test_non_geodataframe_input(self):
        # Test: raises error if input is not GeoDataFrame
        with self.assertRaises(ValueError):
            calculate_risk_direct(pd.DataFrame(), self.raster_path, self.shp_protected, self.shp_frontier)

    def test_valid_geometry_structure(self):
        # Test: ensures test setup GeoDataFrame is valid
        self.assertIn("geometry", self.df_plots.columns)
        self.assertEqual(len(self.df_plots), 1)
        self.assertTrue(self.df_plots.geometry.iloc[0].is_valid)

    def test_default_deforested_value_parameter(self):
        # Test: ensure default deforested_value argument is 2
        import inspect
        params = inspect.signature(calculate_risk_direct).parameters
        self.assertEqual(params['deforested_value'].default, 2)

class TestCalculateRiskDirectSpatial(unittest.TestCase):

    def setUp(self):
        self.plot_geom = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
        self.gdf_plot = gpd.GeoDataFrame({
            "id": [1],
            "geometry": [self.plot_geom]
        }, crs="EPSG:3857")

        self.raster_data = np.zeros((1, 20, 20), dtype=np.uint8)
        self.raster_data[0, 5:10, 5:10] = 2  # deforestation block

        self.transform = from_origin(0, 20, 1, 1)
        self.meta = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'width': 20,
            'height': 20,
            'count': 1,
            'crs': 'EPSG:3857',
            'transform': self.transform
        }

        self.temp_dir = TemporaryDirectory()
        self.raster_path = f"{self.temp_dir.name}/mock_raster.tif"
        with rasterio.open(self.raster_path, 'w', **self.meta) as dst:
            dst.write(self.raster_data)

        protected_geom = [Polygon([(15, 15), (15, 25), (25, 25), (25, 15)])]
        frontier_geom = [Polygon([(0, 0), (0, 30), (30, 30), (30, 0)])]

        self.protected_path = f"{self.temp_dir.name}/protected.shp"
        self.frontier_path = f"{self.temp_dir.name}/frontier.shp"

        gdf_protected = gpd.GeoDataFrame({'id': [1]}, geometry=protected_geom, crs="EPSG:3857")
        gdf_frontier = gpd.GeoDataFrame({'id': [1]}, geometry=frontier_geom, crs="EPSG:3857")

        gdf_protected.to_file(self.protected_path)
        gdf_frontier.to_file(self.frontier_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_risk_calculation_with_mock_data(self):
        result = calculate_risk_direct(
            df_plots=self.gdf_plot,
            raster_deforestation_path=self.raster_path,
            shp_protected_areas_path=self.protected_path,
            shp_farming_areas_path=self.frontier_path,
            deforested_value=2
        )

        self.assertEqual(len(result), 1)
        self.assertIn("risk_direct_level", result.columns)
        self.assertIn("deforested_hectares", result.columns)
        self.assertGreaterEqual(result["deforested_hectares"].iloc[0], 0)
        self.assertIn("risk_context", result.columns)
        self.assertIsInstance(result["risk_context"].iloc[0], dict)

if __name__ == "__main__":
    unittest.main()
