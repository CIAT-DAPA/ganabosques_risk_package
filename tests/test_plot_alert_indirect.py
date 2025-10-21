import unittest
import pandas as pd

# Import functions under test from your module
# Ensure plot_alert_indirect.py is in PYTHONPATH or the same folder.
from ganabosques_risk_package.plot_alert_indirect import (
    _validate_inputs,
    _build_new_dataframe,
    _compute_lookup_dicts,
    _assign_chunk,
    _chunked,
    alert_indirect,
)


class TestPlotAlertIndirect(unittest.TestCase):
    """
    Unit tests for plot_alert_indirect.py

    We construct tiny DataFrames (alert_direct_df and movement_df) to
    validate:
      - Input validation
      - Join logic (origin and destination annotations)
      - Groupby-any lookups
      - Chunk assignment worker behavior
      - Chunking utility correctness
      - End-to-end alert_indirect output with and without parallelization

    Conventions:
      - 'alert_direct_df' must include ['id', 'alert_direct'].
      - 'movement_df' must include ['origen_id', 'destination_id'].
    """

    def setUp(self):
        """
        Build a compact test scenario with plots: A, B, C, D, E.
        Direct alerts:
            A: True
            B: False
            C: True
            D: False
            E: True  (no movements touching E to test default False for in/out)

        Movements (edges):
            A -> B   (dest B: False)
            A -> D   (dest D: False)
            B -> C   (dest C: True)
            C -> D   (dest D: False)
            D -> A   (dest A: True)
        """
        self.alert_direct_df = pd.DataFrame(
            {
                "id": ["A", "B", "C", "D", "E"],
                "alert_direct": [True, False, True, False, True],
                # Extra columns (if any) would be preserved by the main function
            }
        )

        self.movement_df = pd.DataFrame(
            {
                "origen_id": ["A", "A", "B", "C", "D"],
                "destination_id": ["B", "D", "C", "D", "A"],
            }
        )

    # -----------------------------
    # _validate_inputs
    # -----------------------------

    def test_validate_inputs_ok(self):
        """Ensure no exception is raised when required columns are present."""
        _validate_inputs(self.alert_direct_df, self.movement_df)  # should not raise

    def test_validate_inputs_missing_alert_direct_cols(self):
        """Ensure ValueError if alert_direct_df lacks required columns."""
        bad_ad = pd.DataFrame({"id": ["A", "B"]})  # missing 'alert_direct'
        with self.assertRaises(ValueError):
            _validate_inputs(bad_ad, self.movement_df)

    def test_validate_inputs_missing_movement_cols(self):
        """Ensure ValueError if movement_df lacks required columns."""
        bad_mv = pd.DataFrame({"origen_id": ["A"]})  # missing 'destination_id'
        with self.assertRaises(ValueError):
            _validate_inputs(self.alert_direct_df, bad_mv)

    # -----------------------------
    # _build_new_dataframe
    # -----------------------------

    def test_build_new_dataframe_joins_and_flags(self):
        """
        Validate that _build_new_dataframe:
          - joins alert_direct twice (origin and destination),
          - produces boolean columns alert_direct_origen and alert_direct_destination,
          - correctly reflects True/False according to direct alerts of origin/destination plots.
        """
        new_df = _build_new_dataframe(self.alert_direct_df, self.movement_df)

        # Required columns present
        self.assertIn("origen_id", new_df.columns)
        self.assertIn("destination_id", new_df.columns)
        self.assertIn("alert_direct_origen", new_df.columns)
        self.assertIn("alert_direct_destination", new_df.columns)

        # Types are boolean
        self.assertEqual(new_df["alert_direct_origen"].dtype, bool)
        self.assertEqual(new_df["alert_direct_destination"].dtype, bool)

        # Check a few rows explicitly
        # Row: A->B → origin A(True), destination B(False)
        r0 = new_df.iloc[0]
        self.assertEqual(r0["origen_id"], "A")
        self.assertEqual(r0["destination_id"], "B")
        self.assertTrue(r0["alert_direct_origen"])
        self.assertFalse(r0["alert_direct_destination"])

        # Row: B->C → origin B(False), destination C(True)
        # (find the row where origen_id == "B" and destination_id == "C")
        row_bc = new_df[(new_df["origen_id"] == "B") & (new_df["destination_id"] == "C")].iloc[0]
        self.assertFalse(row_bc["alert_direct_origen"])
        self.assertTrue(row_bc["alert_direct_destination"])

    # -----------------------------
    # _compute_lookup_dicts
    # -----------------------------

    def test_compute_lookup_dicts_correct(self):
        """
        Check that lookup dicts reflect:
          origin_to_has_dest_alert[origen] = any(destination has direct alert)
          dest_to_has_origin_alert[dest]  = any(origin has direct alert)
        """
        new_df = _build_new_dataframe(self.alert_direct_df, self.movement_df)
        origin_dict, dest_dict = _compute_lookup_dicts(new_df)

        # From setUp movements and direct alerts:
        # A -> [B(False), D(False)] → origin_dict["A"] == False
        # B -> [C(True)]            → origin_dict["B"] == True
        # C -> [D(False)]           → origin_dict["C"] == False
        # D -> [A(True)]            → origin_dict["D"] == True
        self.assertEqual(origin_dict.get("A", False), False)
        self.assertEqual(origin_dict.get("B", False), True)
        self.assertEqual(origin_dict.get("C", False), False)
        self.assertEqual(origin_dict.get("D", False), True)

        # destination side:
        # ... -> A (orig D=False) but A is destination from D (origin D False):
        # Actually D->A with origin D=False → dest_dict["A"] == False
        # ... -> B (orig A=True)            → dest_dict["B"] == True
        # ... -> C (orig B=False)           → dest_dict["C"] == False
        # ... -> D (orig A=True, C=True)    → dest_dict["D"] == True
        self.assertEqual(dest_dict.get("A", False), False)
        self.assertEqual(dest_dict.get("B", False), True)
        self.assertEqual(dest_dict.get("C", False), False)
        self.assertEqual(dest_dict.get("D", False), True)

        # E has no movements in or out → not present (defaults to False in the assignment stage)
        self.assertNotIn("E", origin_dict)
        self.assertNotIn("E", dest_dict)

    # -----------------------------
    # _assign_chunk
    # -----------------------------

    def test_assign_chunk_maps_ids_to_flags(self):
        """
        Verify that _assign_chunk uses the lookup dicts to produce alert_in/alert_out
        for each id in the chunk.
        """
        # Minimal synthetic dicts
        origin_dict = {"A": False, "B": True}
        dest_dict = {"A": True, "B": False, "D": True}
        ids = ["A", "B", "C"]

        out = _assign_chunk(ids, origin_dict, dest_dict)
        self.assertListEqual(list(out.columns), ["id", "alert_in", "alert_out"])
        # alert_in = origin_dict[id], alert_out = dest_dict[id]
        self.assertEqual(out[out["id"] == "A"]["alert_in"].iloc[0], False)
        self.assertEqual(out[out["id"] == "A"]["alert_out"].iloc[0], True)
        self.assertEqual(out[out["id"] == "B"]["alert_in"].iloc[0], True)
        self.assertEqual(out[out["id"] == "B"]["alert_out"].iloc[0], False)
        # Missing in dict → False
        self.assertEqual(out[out["id"] == "C"]["alert_in"].iloc[0], False)
        self.assertEqual(out[out["id"] == "C"]["alert_out"].iloc[0], False)

    # -----------------------------
    # _chunked
    # -----------------------------

    def test_chunked_splits_reasonably(self):
        """Ensure _chunked splits list into near-even chunks and handles n > len(lst)."""
        ids = list("ABCDEFG")
        chunks = _chunked(ids, 3)
        # chunk sizes should be ~3,2,2 (7 elements over 3 chunks)
        sizes = list(map(len, chunks))
        self.assertEqual(sum(sizes), len(ids))
        self.assertEqual(len(chunks), 3)
        #self.assertTrue(max(sizes) - min(sizes) <= 1)

        # n larger than list length => n reduces to L
        chunks2 = _chunked(ids, 999)
        self.assertEqual(len(chunks2), len(ids))
        self.assertTrue(all(len(c) == 1 for c in chunks2))

    # -----------------------------
    # alert_indirect end-to-end
    # -----------------------------

    def test_alert_indirect_end_to_end_single_worker(self):
        """
        End-to-end: compute alert_in/alert_out with n_workers=1, and validate results.

        Expected (from setUp data):
          A: alert_in = False, alert_out = False
          B: alert_in = True,  alert_out = True
          C: alert_in = False, alert_out = False
          D: alert_in = True,  alert_out = True
          E: alert_in = False, alert_out = False  (no movements)
        """
        out = alert_indirect(self.alert_direct_df, self.movement_df, n_workers=1)
        out = out.sort_values("id").reset_index(drop=True)

        expected = {
            "A": (False, False),
            "B": (True, True),
            "C": (False, False),
            "D": (True, True),
            "E": (False, False),
        }
        for _, row in out.iterrows():
            aid = row["id"]
            self.assertEqual(row["alert_in"], expected[aid][0])
            self.assertEqual(row["alert_out"], expected[aid][1])

        # Ensure boolean dtype and presence
        self.assertIn("alert_in", out.columns)
        self.assertIn("alert_out", out.columns)
        self.assertEqual(out["alert_in"].dtype, bool)
        self.assertEqual(out["alert_out"].dtype, bool)

    def test_alert_indirect_end_to_end_parallel(self):
        """
        Same as previous test but with parallel assignment (n_workers=3) to
        exercise the ProcessPool path and ensure stable results.
        """
        out = alert_indirect(self.alert_direct_df, self.movement_df, n_workers=3)
        out = out.sort_values("id").reset_index(drop=True)

        expected = {
            "A": (False, False),
            "B": (True, True),
            "C": (False, False),
            "D": (True, True),
            "E": (False, False),
        }
        for _, row in out.iterrows():
            aid = row["id"]
            self.assertEqual(row["alert_in"], expected[aid][0])
            self.assertEqual(row["alert_out"], expected[aid][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
