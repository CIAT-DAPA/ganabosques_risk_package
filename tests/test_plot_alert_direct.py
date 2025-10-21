# Filename: tests/test_plot_alert_direct_files_unittest.py
# How to run:
#   python -m unittest -v tests/test_plot_alert_direct_files_unittest.py

import os
import math
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd

# Import the function under test
from ganabosques_risk_package.plot_alert_direct import alert_direct


class TestPlotAlertDirect(unittest.TestCase):
    """
    End-to-end tests for alert_direct() using real fixtures located in tests/data_test:
      - areas.shp                -> used for both protected_areas and farming_areas
      - plot_intersect.shp       -> a plot intersecting areas and deforestation
      - plot_not_intersect.shp   -> a plot not intersecting areas nor deforestation
      - deforestation.tif        -> deforestation raster
    """

    @classmethod
    def setUpClass(cls):
        # Resolve data_test directory relative to this test file
        cls.repo_root = Path(__file__).resolve().parents[1]  # assumes tests/ at repo root
        cls.data_dir = cls.repo_root / "tests" / "data_test"

        # Paths
        cls.areas_path = str(cls.data_dir / "areas" / "areas.shp")
        cls.plot_intersect_path = str(cls.data_dir / "plot_intersect" / "plot_intersect.shp")
        cls.plot_not_intersect_path = str(cls.data_dir / "plot_not_intersect" / "plot_not_intersect.shp")
        cls.deforestation_path = str(cls.data_dir / "deforestation.tif")

        # Load vectors
        cls.areas_gdf = gpd.read_file(cls.areas_path)
        cls.gdf_intersect = gpd.read_file(cls.plot_intersect_path)
        cls.gdf_not_intersect = gpd.read_file(cls.plot_not_intersect_path)

        # Ensure an 'id' column for plots if not present
        if "id" not in cls.gdf_intersect.columns:
            cls.gdf_intersect = cls.gdf_intersect.copy()
            cls.gdf_intersect["id"] = [
                f"plot_intersect_{i}" for i in range(len(cls.gdf_intersect))
            ]
        if "id" not in cls.gdf_not_intersect.columns:
            cls.gdf_not_intersect = cls.gdf_not_intersect.copy()
            cls.gdf_not_intersect["id"] = [
                f"plot_not_intersect_{i}" for i in range(len(cls.gdf_not_intersect))
            ]

        # Basic existence checks
        for p in [cls.areas_path, cls.plot_intersect_path, cls.plot_not_intersect_path, cls.deforestation_path]:
            if not Path(p).exists():
                raise FileNotFoundError(f"Required test fixture not found: {p}")

    # --------------- Helpers ---------------

    def _run_alert_direct_single(self, plots_gdf: gpd.GeoDataFrame, n_workers: int = 1) -> pd.Series:
        """
        Run alert_direct for a (possibly multi-row) plots GeoDataFrame.
        Returns the first row of the resulting DataFrame for convenience.
        """
        out_df = alert_direct(
            plots=plots_gdf,
            deforestation=self.deforestation_path,
            protected_areas=self.areas_path,
            farming_areas=self.areas_path,  # using the same 'areas' as farming areas per spec
            deforestation_value=2,
            n_workers=n_workers,
            id_column="id",
        )
        self.assertGreaterEqual(len(out_df), 1, "alert_direct returned an empty DataFrame")
        return out_df.iloc[0]

    # --------------- Tests: plot that DOES intersect ---------------

    def test_intersect_end_to_end_single_worker(self):
        """
        Purpose:
          - For a plot that intersects with deforestation and areas:
              * deforested_area > 0  -> alert_direct == True
              * protected_areas_area >= 0 (likely > 0)
              * farming_in_area >= 0 and farming_out_area >= 0
              * proportions are consistent with areas and bounded [0,1]
        """
        row = self._run_alert_direct_single(self.gdf_intersect.head(1), n_workers=1)

        # Basic presence of expected columns
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
        self.assertTrue(all(c in row.index for c in expected_cols))

        # Assertions for intersecting plot
        self.assertGreater(row["plot_area"], 0.0)
        self.assertGreater(row["deforested_area"], 0.0)
        self.assertTrue(row["alert_direct"])

        # Areas must be non-negative
        self.assertGreaterEqual(row["protected_areas_area"], 0.0)
        self.assertGreaterEqual(row["farming_in_area"], 0.0)
        self.assertGreaterEqual(row["farming_out_area"], 0.0)

        # Farming in/out must partition the plot (with tolerance)
        self.assertTrue(
            math.isclose(
                float(row["farming_in_area"]) + float(row["farming_out_area"]),
                float(row["plot_area"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ),
            "farming_in_area + farming_out_area must equal plot_area (within tolerance)",
        )

        # Proportions bounded in [0,1]
        for prop_col, area_col in [
            ("deforested_proportion", "deforested_area"),
            ("protected_areas_proportion", "protected_areas_area"),
            ("farming_in_proportion", "farming_in_area"),
            ("farming_out_proportion", "farming_out_area"),
        ]:
            prop = float(row[prop_col])
            self.assertGreaterEqual(prop, 0.0, f"{prop_col} must be >= 0")
            self.assertLessEqual(prop, 1.0, f"{prop_col} must be <= 1")
            # check proportionality: prop ≈ area / plot_area (when plot_area > 0)
            self.assertTrue(
                math.isclose(
                    prop,
                    float(row[area_col]) / float(row["plot_area"]) if row["plot_area"] > 0 else 0.0,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ),
                f"{prop_col} must equal {area_col}/plot_area within tolerance",
            )

    def test_intersect_end_to_end_parallel(self):
        """
        Purpose:
          - Same as the previous test but with parallel path (n_workers=2)
          - Ensures deterministic results with multiprocessing
        """
        row = self._run_alert_direct_single(self.gdf_intersect.head(1), n_workers=2)
        # Key signals
        self.assertGreater(row["deforested_area"], 0.0)
        self.assertTrue(row["alert_direct"])
        # Partition check
        self.assertTrue(
            math.isclose(
                float(row["farming_in_area"]) + float(row["farming_out_area"]),
                float(row["plot_area"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )

    # --------------- Tests: plot that DOES NOT intersect ---------------

    def test_not_intersect_end_to_end_single_worker(self):
        """
        Purpose:
          - For a plot that does not intersect areas nor deforestation:
              * deforested_area == 0 -> alert_direct == False
              * protected_areas_area == 0
              * farming_in_area == 0 and farming_out_area == plot_area
              * all proportions == 0 except farming_out_proportion == 1
        """
        row = self._run_alert_direct_single(self.gdf_not_intersect.head(1), n_workers=1)

        self.assertGreater(row["plot_area"], 0.0)

        # Deforestation & protected areas
        self.assertEqual(float(row["deforested_area"]), 0.0)
        self.assertFalse(row["alert_direct"])
        self.assertEqual(float(row["protected_areas_area"]), 0.0)

        # Farming partition: all outside
        self.assertEqual(float(row["farming_in_area"]), 0.0)
        self.assertTrue(
            math.isclose(
                float(row["farming_out_area"]),
                float(row["plot_area"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            ),
            "For non-intersecting plot, farming_out_area should equal plot_area",
        )

        # Proportions
        self.assertEqual(float(row["deforested_proportion"]), 0.0)
        self.assertEqual(float(row["protected_areas_proportion"]), 0.0)
        self.assertEqual(float(row["farming_in_proportion"]), 0.0)
        self.assertTrue(
            math.isclose(
                float(row["farming_out_proportion"]),
                1.0,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ),
            "For non-intersecting plot, farming_out_proportion should be 1.0",
        )

    def test_not_intersect_end_to_end_parallel(self):
        """
        Purpose:
          - Same as previous test but with n_workers=2 to exercise parallel branch.
        """
        row = self._run_alert_direct_single(self.gdf_not_intersect.head(1), n_workers=2)
        self.assertEqual(float(row["deforested_area"]), 0.0)
        self.assertFalse(row["alert_direct"])
        self.assertEqual(float(row["farming_in_area"]), 0.0)
        self.assertTrue(
            math.isclose(
                float(row["farming_out_area"]),
                float(row["plot_area"]),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )
        self.assertTrue(math.isclose(float(row["farming_out_proportion"]), 1.0, rel_tol=1e-9, abs_tol=1e-9))


if __name__ == "__main__":
    unittest.main(verbosity=2)
