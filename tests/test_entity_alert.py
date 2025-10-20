import unittest
import pandas as pd

# Import functions from your module (ensure entity_alert.py is in PYTHONPATH or same folder)
from ganabosques_risk_package.entity_alert import (
    _validate_alert_indirect,
    _normalize_alert_indirect,
    _normalize_entity,
    _normalize_provider,
    _chunk,
    _aggregate_chunk,
    calculate_alert,
)


class TestEntityAlert(unittest.TestCase):
    """
    Unit tests for entity_alert.py.

    This suite verifies:
      - Input validation: required columns and clear error messages.
      - Normalization: column name coercion and dtype coercion.
      - Provider normalization: column detection and deduping.
      - Chunk splitting: balanced contiguous chunks and edge cases.
      - Per-chunk aggregation: exact counts/sums and final 'alert' logic.
      - End-to-end 'calculate_alert': single worker and parallel workers.
      - Edge cases: entities without plots, empty provider, non-existent plots in provider, duplicates, many-to-many mapping.
    """

    def setUp(self):
        """
        Build minimal, deterministic inputs.

        alert_indirect_df:
            id  deforested_area  alert_direct  alert_in  alert_out
           101        0.5             True       False     False
           102        1.2             False      True      False
           103        0.0             False      False     True
           104        3.3             True       False     False
           105       "0.0"            False      False     False   (string -> coerced)
           106        None            False      False     False   (None -> 0.0)

        entity_df: master entities [X,Y,Z,W]
            id, name  -> normalize to entity_id, entity_name

        provider_df: mapping plot -> entity
            X: 101, 102
            Y: 103, 104
            Z: 105
            Z: 999 (non-existent plot in alert_indirect_df; should be safely handled as zeros/False)
            (Also duplicate mapping 102->X to verify deduplication)
        """
        self.alert_indirect_df = pd.DataFrame({
            "id": [101, 102, 103, 104, 105, 106],
            "deforested_area": [0.5, 1.2, 0.0, 3.3, "0.0", None],
            "alert_direct": [True, False, False, True, False, False],
            "alert_in": [False, True, False, False, False, False],
            "alert_out": [False, False, True, False, False, False],
        })

        self.entity_df = pd.DataFrame({
            "id": ["W", "X", "Y", "Z"],
            "name": ["NoPlots", "Alpha", "Beta", "Gamma"]
        })

        # Provider with duplicate (102, X) to be deduped
        self.provider_df = pd.DataFrame({
            "plot_id": [101, 102, 102, 103, 104, 105, 999],
            "entity_id": ["X",  "X",  "X",  "Y",  "Y",  "Z",  "Z"],
        })

        # Expected results per entity when using the above inputs
        # X: plots = {101,102} -> totals=2
        #    direct: True (101), False (102) -> 1
        #    in:  False (101), True (102)   -> 1
        #    out: False, False              -> 0
        #    def_area: 0.5 + 1.2 = 1.7
        #    alert: True (direct or in/out)
        # Y: plots = {103,104} -> totals=2
        #    direct: False, True -> 1
        #    in:     False, False -> 0
        #    out:    True, False -> 1
        #    def_area: 0.0 + 3.3 = 3.3
        #    alert: True
        # Z: plots = {105, 999}
        #    105 exists (direct False, in False, out False, area "0.0"->0.0)
        #    999 does not exist -> filled with zeros/False
        #    totals=2 (because provider has two rows for Z)
        #    direct=0, in=0, out=0, def_area=0.0, alert=False
        # W: no provider rows -> totals=0, all zeros/False
        self.expected = {
            "W": dict(plots_total=0, plots_alert_direct=0, plots_alert_in=0, plots_alert_out=0,
                      deforested_area_sum=0.0, alert=False),
            "X": dict(plots_total=2, plots_alert_direct=1, plots_alert_in=1, plots_alert_out=0,
                      deforested_area_sum=1.7, alert=True),
            "Y": dict(plots_total=2, plots_alert_direct=1, plots_alert_in=0, plots_alert_out=1,
                      deforested_area_sum=3.3, alert=True),
            "Z": dict(plots_total=2, plots_alert_direct=0, plots_alert_in=0, plots_alert_out=0,
                      deforested_area_sum=0.0, alert=False),
        }

    # ----------------------------------
    # Validation
    # ----------------------------------

    def test_validate_alert_indirect_ok(self):
        """No error when alert_indirect has the required columns."""
        _validate_alert_indirect(self.alert_indirect_df)  # must not raise

    def test_validate_alert_indirect_missing_columns(self):
        """ValueError when any required column is missing."""
        bad = self.alert_indirect_df.drop(columns=["alert_out"])
        with self.assertRaises(ValueError):
            _validate_alert_indirect(bad)

    # ----------------------------------
    # Normalization
    # ----------------------------------

    def test_normalize_alert_indirect_types(self):
        """
        Coerce deforested_area to numeric (strings/None -> 0.0)
        and boolean flags to bool dtype.
        """
        norm = _normalize_alert_indirect(self.alert_indirect_df)
        # deforested_area coercion
        self.assertEqual(norm.loc[norm["id"] == 105, "deforested_area"].iloc[0], 0.0)
        self.assertEqual(norm.loc[norm["id"] == 106, "deforested_area"].iloc[0], 0.0)
        # bool dtypes
        self.assertEqual(norm["alert_direct"].dtype, bool)
        self.assertEqual(norm["alert_in"].dtype, bool)
        self.assertEqual(norm["alert_out"].dtype, bool)

    def test_normalize_entity_variants(self):
        """
        Accept 'id'/'name' variants -> normalize to entity_id/entity_name,
        keep only required columns and unique rows.
        """
        master = _normalize_entity(self.entity_df)
        self.assertListEqual(list(master.columns), ["entity_id", "entity_name"])
        self.assertSetEqual(set(master["entity_id"]), set(["W", "X", "Y", "Z"]))
        names = dict(zip(master["entity_id"], master["entity_name"]))
        self.assertEqual(names["X"], "Alpha")
        self.assertEqual(names["W"], "NoPlots")

    def test_normalize_entity_missing(self):
        """Raise ValueError when both id or name variants are missing."""
        with self.assertRaises(ValueError):
            _normalize_entity(pd.DataFrame({"name": ["A"]}))  # missing id
        with self.assertRaises(ValueError):
            _normalize_entity(pd.DataFrame({"id": ["A"]}))    # missing name

    def test_normalize_provider_variants_and_dedup(self):
        """
        Normalize provider to plot_id/entity_id and drop duplicate rows.
        Ensure duplicates are removed (e.g., (102, X) appears once).
        """
        norm = _normalize_provider(self.provider_df)
        self.assertListEqual(list(norm.columns), ["plot_id", "entity_id"])
        # Count rows after dedup: original had 7 rows, one duplicate (102,X) -> expect 6
        self.assertEqual(len(norm), 6)
        # Ensure (102, X) appears exactly once
        dup = norm[(norm["plot_id"] == 102) & (norm["entity_id"] == "X")]
        self.assertEqual(len(dup), 1)

    def test_normalize_provider_missing_cols(self):
        """Raise ValueError when neither plot id nor entity id columns are found."""
        with self.assertRaises(ValueError):
            _normalize_provider(pd.DataFrame({"entity_id": ["X"]}))  # no plot id
        with self.assertRaises(ValueError):
            _normalize_provider(pd.DataFrame({"plot_id": [1]}))      # no entity id

    # ----------------------------------
    # Chunking helper
    # ----------------------------------

    def test_chunk_balanced(self):
        """Split into ~balanced contiguous chunks and handle n > len."""
        values = list("ABCDEFG")
        chunks = _chunk(values, n_chunks=3)
        sizes = [len(c) for c in chunks]
        self.assertEqual(sum(sizes), len(values))
        self.assertEqual(len(chunks), 3)
        self.assertTrue(max(sizes) - min(sizes) <= 1)

        chunks2 = _chunk(values, n_chunks=100)
        self.assertEqual(len(chunks2), len(values))
        self.assertTrue(all(len(c) == 1 for c in chunks2))

    # ----------------------------------
    # Per-chunk aggregation
    # ----------------------------------

    def test_aggregate_chunk_metrics(self):
        """
        Aggregate a subset of entity_ids and match exact expected metrics.
        We simulate the merged table as produced by calculate_alert.
        """
        # Build merged-like table: provider × plots (with coercions)
        norm_plots = _normalize_alert_indirect(self.alert_indirect_df)
        norm_provider = _normalize_provider(self.provider_df)
        merged = norm_provider.merge(norm_plots, left_on="plot_id", right_on="id", how="left")
        merged["deforested_area"] = pd.to_numeric(merged["deforested_area"], errors="coerce").fillna(0.0)
        for col in ["alert_direct", "alert_in", "alert_out"]:
            merged[col] = merged[col].fillna(False).astype(bool)

        # Aggregate for X and Y only
        chunk_df = _aggregate_chunk(merged, entity_ids=["X", "Y"])
        self.assertSetEqual(set(chunk_df["entity_id"]), set(["X", "Y"]))

        # Compare with expectations
        for eid in ["X", "Y"]:
            row = chunk_df[chunk_df["entity_id"] == eid].iloc[0]
            exp = self.expected[eid]
            self.assertEqual(row["plots_total"], exp["plots_total"])
            self.assertEqual(row["plots_alert_direct"], exp["plots_alert_direct"])
            self.assertEqual(row["plots_alert_in"], exp["plots_alert_in"])
            self.assertEqual(row["plots_alert_out"], exp["plots_alert_out"])
            self.assertAlmostEqual(row["deforested_area_sum"], exp["deforested_area_sum"], places=7)
            self.assertEqual(row["alert"], exp["alert"])

    # ----------------------------------
    # End-to-end calculate_alert
    # ----------------------------------

    def _assert_end_to_end(self, n_workers: int):
        """Helper to run calculate_alert and verify the final table."""
        out = calculate_alert(
            alert_indirect_df=self.alert_indirect_df,
            entity_df=self.entity_df,
            provider_df=self.provider_df,
            n_workers=n_workers,
        )
        # Expected stable schema and entity order (sorted by entity_id)
        expected_cols = [
            "entity_id",
            "entity_name",
            "plots_total",
            "plots_alert_direct",
            "plots_alert_in",
            "plots_alert_out",
            "deforested_area_sum",
            "alert",
        ]
        self.assertListEqual(list(out.columns), expected_cols)
        self.assertListEqual(out["entity_id"].tolist(), ["W", "X", "Y", "Z"])

        # Validate every entity row against the expectations
        for _, row in out.iterrows():
            eid = row["entity_id"]
            exp = self.expected[eid]
            self.assertEqual(row["plots_total"], exp["plots_total"])
            self.assertEqual(row["plots_alert_direct"], exp["plots_alert_direct"])
            self.assertEqual(row["plots_alert_in"], exp["plots_alert_in"])
            self.assertEqual(row["plots_alert_out"], exp["plots_alert_out"])
            self.assertAlmostEqual(row["deforested_area_sum"], exp["deforested_area_sum"], places=7)
            self.assertEqual(row["alert"], exp["alert"])

        # Dtypes sanity checks
        self.assertTrue(pd.api.types.is_integer_dtype(out["plots_total"]))
        self.assertTrue(pd.api.types.is_integer_dtype(out["plots_alert_direct"]))
        self.assertTrue(pd.api.types.is_integer_dtype(out["plots_alert_in"]))
        self.assertTrue(pd.api.types.is_integer_dtype(out["plots_alert_out"]))
        self.assertTrue(pd.api.types.is_float_dtype(out["deforested_area_sum"]))
        self.assertTrue(pd.api.types.is_bool_dtype(out["alert"]))

    def test_calculate_alert_single_worker(self):
        """End-to-end with single worker (n_workers=1)."""
        self._assert_end_to_end(n_workers=1)

    def test_calculate_alert_parallel(self):
        """End-to-end with parallel workers (n_workers>1) to exercise ProcessPool path."""
        self._assert_end_to_end(n_workers=3)

    # ----------------------------------
    # Edge cases
    # ----------------------------------

    def test_empty_provider(self):
        """
        Provider is empty -> every entity should return zeros/False.
        Ensures robustness when there is no mapping (e.g., filters applied upstream).
        """
        empty_provider = pd.DataFrame({"plot_id": [], "entity_id": []})
        out = calculate_alert(self.alert_indirect_df, self.entity_df, empty_provider, n_workers=2)
        self.assertListEqual(out["entity_id"].tolist(), ["W", "X", "Y", "Z"])
        self.assertTrue((out["plots_total"] == 0).all())
        self.assertTrue((out["plots_alert_direct"] == 0).all())
        self.assertTrue((out["plots_alert_in"] == 0).all())
        self.assertTrue((out["plots_alert_out"] == 0).all())
        self.assertTrue((out["deforested_area_sum"] == 0.0).all())
        self.assertTrue((out["alert"] == False).all())

    def test_many_to_many_provider(self):
        """
        Many-to-many mapping: a single plot belongs to multiple entities.
        Verify that counts/sums correctly reflect duplicated inclusion via provider.
        """
        # Clone provider and add extra mapping: plot 101 belongs to Z as well
        mm_provider = pd.concat(
            [self.provider_df, pd.DataFrame({"plot_id": [101], "entity_id": ["Z"]})],
            ignore_index=True,
        )
        out = calculate_alert(self.alert_indirect_df, self.entity_df, mm_provider, n_workers=1)

        # For Z, plots_total should increase by 1; deforested area increases by 0.5
        zrow = out[out["entity_id"] == "Z"].iloc[0]
        self.assertEqual(zrow["plots_total"], self.expected["Z"]["plots_total"] + 1)
        self.assertAlmostEqual(zrow["deforested_area_sum"], self.expected["Z"]["deforested_area_sum"] + 0.5, places=7)

    def test_provider_points_to_nonexistent_plot(self):
        """
        Provider references non-existent plot (999) -> should be filled as zeros/False,
        contributing 1 to plots_total but 0 to metrics.
        """
        out = calculate_alert(self.alert_indirect_df, self.entity_df, self.provider_df, n_workers=1)
        zrow = out[out["entity_id"] == "Z"].iloc[0]
        self.assertEqual(zrow["plots_total"], 2)  # 105 and 999
        self.assertAlmostEqual(zrow["deforested_area_sum"], 0.0, places=7)
        self.assertFalse(zrow["alert"])

    def test_entity_with_no_provider_rows(self):
        """Entity 'W' has no rows in provider -> outputs zeros and alert=False."""
        out = calculate_alert(self.alert_indirect_df, self.entity_df, self.provider_df, n_workers=1)
        wrow = out[out["entity_id"] == "W"].iloc[0]
        self.assertEqual(wrow["plots_total"], 0)
        self.assertFalse(wrow["alert"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
