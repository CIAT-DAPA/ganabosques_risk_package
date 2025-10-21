import math
import os
import tempfile
import unittest
from pathlib import Path
from multiprocessing import shared_memory

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

# Import functions under test (adjust module name if needed)
from ganabosques_risk_package.plot_alert_direct import (
    _init_worker,
    _get_raster_array_view,
    _bounds_to_window,
    _intersect_raster_deforestation,
    _intersect_area_layer,
    _process_one,
    alert_direct,
)


class TestPlotOverlays(unittest.TestCase):
    """
    Test suite for the plot_overlays module using Python's built-in unittest.

    We use a tiny 5x5 raster with 10 m pixels (100 m² = 0.01 ha per pixel) and a set
    of simple polygons carefully aligned to pixel edges so results are exact and
    deterministic. This keeps tests fast and easy to reason about.
    """

    def setUp(self):
        """
        Create synthetic data used across tests:
        - 5x5 uint8 raster with two pixels = 2 (deforestation class)
          at (row=2, col=1) and (row=2, col=2).
        - Transform: from_origin(0, 50, 10, 10) → 10 m pixels, top-left (0, 50)
        - Polygons:
            * plot_poly covers exactly the two 'deforestation' pixels (x:10..30, y:20..30)
            * protected_poly covers the left deforestation pixel (x:10..20, y:20..30)
            * farming_poly fully covers the plot polygon (x:9..31, y:19..31)
        """
        # --- Raster ---
        self.arr = np.zeros((5, 5), dtype=np.uint8)
        self.arr[2, 1] = 2
        self.arr[2, 2] = 2
        self.transform = from_origin(0, 50, 10, 10)  # pixel size 10x10 m
        self.pixel_area_ha = (10 * 10) / 10000.0     # 100 m² = 0.01 ha
        self.nodata = None  # no NoData for simplicity

        # --- Polygons (in same projected-like units) ---
        self.plot_poly = box(10, 20, 30, 30)        # exactly 2 pixels (2*100 m² = 200 m²)
        self.protected_poly = box(10, 20, 20, 30)   # exactly 1 pixel within plot
        self.farming_poly = box(9, 19, 31, 31)      # fully covers plot

        # Build quick GeoDataFrames (no CRS needed here; we keep it None)
        self.protected_gdf = gpd.GeoDataFrame(geometry=[self.protected_poly], crs=None)
        self.farming_gdf = gpd.GeoDataFrame(geometry=[self.farming_poly], crs=None)

    # -------------------------------------------------------------------------
    # Helper: initialize worker globals & shared memory for raster once per test
    # -------------------------------------------------------------------------

    def _init_worker_with_shared_raster(self, protected_gdf=None, farming_gdf=None):
        """
        Utility to place the tiny raster in shared memory and initialize worker
        globals via _init_worker. Returns the shared memory handle so the caller
        can close/unlink it in a 'finally' block.

        Branches/ifs exercised:
        - If NoData is None, valid mask becomes all True inside _intersect_raster_deforestation.
        - If protected/farming layers are empty, sindex building is skipped by the if.
        """
        shm = shared_memory.SharedMemory(create=True, size=self.arr.nbytes)
        shm_arr = np.ndarray(self.arr.shape, dtype=self.arr.dtype, buffer=shm.buf)
        shm_arr[:] = self.arr

        _init_worker(
            shm_name=shm.name,
            raster_shape=self.arr.shape,
            raster_dtype=str(self.arr.dtype),
            raster_nodata=self.nodata,
            affine_params=(
                self.transform.a, self.transform.b, self.transform.c,
                self.transform.d, self.transform.e, self.transform.f
            ),
            pixel_area_ha=self.pixel_area_ha,
            defo_value=2,
            protected_gdf=(protected_gdf if protected_gdf is not None else gpd.GeoDataFrame(geometry=[])),
            farming_gdf=(farming_gdf if farming_gdf is not None else gpd.GeoDataFrame(geometry=[])),
        )
        return shm

    # -----------------------------
    # Tests for internal helpers
    # -----------------------------

    def test_init_worker_and_get_raster_view(self):
        """
        Purpose:
            Validate that _init_worker sets global state correctly and that
            _get_raster_array_view returns a view into shared memory matching
            the original array (shape, dtype, values).

        Steps:
            1) Put raster into shared memory and init worker globals.
            2) Fetch view and compare to original.

        Expected:
            - View shape == (5,5), dtype == uint8, values identical.
        """
        shm = self._init_worker_with_shared_raster()
        try:
            view = _get_raster_array_view()
            self.assertEqual(view.shape, self.arr.shape)
            self.assertEqual(view.dtype, self.arr.dtype)
            np.testing.assert_array_equal(view, self.arr)
        finally:
            shm.close()
            shm.unlink()

    def test_bounds_to_window_inside(self):
        """
        Purpose:
            Ensure _bounds_to_window returns a non-empty, in-bounds window when
            the requested bounds are clearly inside the raster extent.

        Steps:
            - Request a 2x2 pixel area (x:10..30, y:20..40).

        Expected:
            - Positive height/width and offsets within [0, 5).
        """
        bounds = (10, 20, 30, 40)
        row_off, col_off, h, w = _bounds_to_window(bounds, self.transform, raster_width=5, raster_height=5)
        self.assertGreater(h, 0)
        self.assertGreater(w, 0)
        self.assertTrue(0 <= row_off < 5)
        self.assertTrue(0 <= col_off < 5)

    def test_intersect_raster_deforestation_counts_pixels(self):
        """
        Purpose:
            Verify that _intersect_raster_deforestation correctly counts pixels
            with class '2' inside plot_poly and converts to hectares.

        Steps:
            1) Init worker with shared raster.
            2) Call function using plot_poly covering exactly two '2' pixels.

        Expected:
            - Area = 2 * 0.01 ha = 0.02 ha (within tight tolerance).
        """
        shm = self._init_worker_with_shared_raster()
        try:
            view = _get_raster_array_view()
            area_ha = _intersect_raster_deforestation(self.plot_poly, view)
            self.assertTrue(math.isclose(area_ha, 2 * self.pixel_area_ha, rel_tol=1e-6))
        finally:
            shm.close()
            shm.unlink()

    def test_intersect_area_layer_returns_area(self):
        """
        Purpose:
            Confirm that _intersect_area_layer returns the correct overlap area
            (in hectares) between a polygon and a vector layer.

        Steps:
            - Intersect plot_poly with protected_gdf (1 pixel overlap).

        Expected:
            - Area equals 0.01 ha (100 m²).
        """
        area_ha = _intersect_area_layer(self.plot_poly, self.protected_gdf)
        expected = (self.plot_poly.intersection(self.protected_poly).area) / 10000.0
        self.assertTrue(math.isclose(area_ha, expected, rel_tol=1e-6))

    def test_process_one_returns_expected_keys_and_alert(self):
        """
        Purpose:
            Exercise _process_one end-to-end (with globals already initialized):
            - Validates presence of all required keys, including 'alert_direct'.
            - Checks numeric correctness of areas and proportions.
            - Validates 'if' branches for zero-division guards and area clamping.

        Steps:
            1) Init worker with raster + both vector layers.
            2) Build record (id, WKB).
            3) Call _process_one and assert fields.

        Expected:
            - deforested_area = 0.02 ha, alert_direct = True
            - farming_in_area == plot_area, farming_out_area == 0
            - protected_areas_area = 0.01 ha
        """
        shm = self._init_worker_with_shared_raster(
            protected_gdf=self.protected_gdf, farming_gdf=self.farming_gdf
        )
        try:
            record = ("plot-1", self.plot_poly.wkb)
            out = _process_one(record)

            # Keys exist?
            expected_keys = {
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
            }
            self.assertTrue(expected_keys.issubset(set(out.keys())))

            # Plot area = 200 m² = 0.02 ha
            self.assertTrue(
                math.isclose(out["plot_area"], self.plot_poly.area / 10000.0, rel_tol=1e-6)
            )

            # Deforestation = 2 pixels → 0.02 ha; alert is True when area > 0
            self.assertTrue(math.isclose(out["deforested_area"], 2 * self.pixel_area_ha, rel_tol=1e-6))
            self.assertTrue(out["alert_direct"])

            # Farming fully covers plot → in == plot_area; out == 0
            self.assertTrue(math.isclose(out["farming_in_area"], out["plot_area"], rel_tol=1e-6))
            self.assertTrue(math.isclose(out["farming_out_area"], 0.0, rel_tol=1e-9))

            # Protected area = 1 pixel → 0.01 ha
            self.assertTrue(math.isclose(out["protected_areas_area"], 1 * self.pixel_area_ha, rel_tol=1e-6))
        finally:
            shm.close()
            shm.unlink()

    # -------------------------------------------
    # End-to-end test for alert_direct
    # -------------------------------------------

    def test_alert_direct_end_to_end(self):
        """
        Purpose:
            Validate full pipeline with actual files on disk:
            - Write GeoTIFF raster and GeoJSON vector layers to temp dir.
            - Build plots GeoDataFrame with one polygon.
            - Run alert_direct(n_workers=1) to avoid multiprocessing flakiness.
            - Check columns and key numeric results, including 'alert_direct'.

        Branches/ifs covered:
            - Vector layers are non-empty → sindex initialization path.
            - Proportion calculations with non-zero plot area.
            - Final DataFrame column ordering.

        Expected:
            - deforested_area = 0.02 ha
            - protected_areas_area = 0.01 ha
            - farming_in_area == plot_area
            - farming_out_area == 0
            - alert_direct == True
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raster_path = tmpdir / "deforestation.tif"
            protected_path = tmpdir / "protected.geojson"
            farming_path = tmpdir / "farming.geojson"

            # --- Write raster to GeoTIFF ---
            with rasterio.open(
                raster_path,
                "w",
                driver="GTiff",
                height=self.arr.shape[0],
                width=self.arr.shape[1],
                count=1,
                dtype=self.arr.dtype,
                crs=None,  # keep None; function will "reproject" vectors to this CRS (noop when None)
                transform=self.transform,
                nodata=self.nodata,
            ) as dst:
                dst.write(self.arr, 1)

            # --- Write vector layers to GeoJSON ---
            self.protected_gdf.to_file(protected_path, driver="GeoJSON")
            self.farming_gdf.to_file(farming_path, driver="GeoJSON")

            # --- Build plots GeoDataFrame ---
            plots = gpd.GeoDataFrame({"id": ["plot-1"], "geometry": [self.plot_poly]}, crs=None)

            # --- Run the function (single worker for simplicity/testing) ---
            df = alert_direct(
                plots=plots,
                deforestation=str(raster_path),
                protected_areas=str(protected_path),
                farming_areas=str(farming_path),
                deforestation_value=2,
                n_workers=1,
                id_column="id",
            )

            # --- Validate schema (including alert_direct) ---
            expected_cols = [
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
            self.assertEqual(list(df.columns), expected_cols)
            self.assertEqual(df.shape[0], 1)

            # --- Validate values ---
            row = df.iloc[0]
            self.assertTrue(math.isclose(row["deforested_area"], 2 * self.pixel_area_ha, rel_tol=1e-6))
            self.assertTrue(row["alert_direct"])

            self.assertTrue(math.isclose(row["protected_areas_area"], 1 * self.pixel_area_ha, rel_tol=1e-6))
            self.assertTrue(math.isclose(row["farming_in_area"], row["plot_area"], rel_tol=1e-6))
            self.assertTrue(math.isclose(row["farming_out_area"], 0.0, rel_tol=1e-9))

            # Proportions should be in [0,1] (allow tiny numerical slop)
            for col in [
                "deforested_proportion",
                "protected_areas_proportion",
                "farming_in_proportion",
                "farming_out_proportion",
            ]:
                self.assertGreaterEqual(row[col], -1e-12)
                self.assertLessEqual(row[col], 1.0 + 1e-9)


if __name__ == "__main__":
    # Running via `python test_plot_overlays_unittest.py` will trigger this.
    unittest.main(verbosity=2)
