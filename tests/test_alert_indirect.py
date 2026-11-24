import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from ganabosques_risk_package.alert_indirect import alert_indirect


def run(*args, **kwargs) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        return alert_indirect(*args, **kwargs)


class TestAlertIndirect(unittest.TestCase):

    def setUp(self):
        # A tiene alerta directa. Movimientos: A->B, B->C, C->A, D->B
        self.direct = pd.DataFrame({
            "id": ["A", "B", "C", "D", "E"],
            "direct_alert": [True, False, False, False, False],
        })
        self.mov = pd.DataFrame({
            "origen_id": ["A", "B", "C", "D"],
            "destination_id": ["B", "C", "A", "B"],
        })

    def by_id(self, df):
        return df.set_index("id")

    def test_output_columns(self):
        out = run(self.direct, self.mov)
        self.assertEqual(list(out.columns), [
            "id", "n_in", "n_out", "n_indirect_in", "n_indirect_out",
            "n_total_mov", "indirect_alert_in", "indirect_alert_out",
        ])
        self.assertEqual(out["id"].tolist(), ["A", "B", "C", "D", "E"])

    def test_semantics(self):
        out = self.by_id(run(self.direct, self.mov))
        # B recibe de A (alertado) -> indirect_alert_in
        self.assertTrue(out.loc["B", "indirect_alert_in"])
        self.assertEqual(out.loc["B", "n_in"], 2)
        self.assertEqual(out.loc["B", "n_indirect_in"], 1)
        # C envía a A (alertado) -> indirect_alert_out
        self.assertTrue(out.loc["C", "indirect_alert_out"])
        self.assertFalse(out.loc["C", "indirect_alert_in"])
        # D envía a B (no alertado)
        self.assertFalse(out.loc["D", "indirect_alert_out"])
        # E no tiene movimientos
        self.assertEqual(out.loc["E", "n_total_mov"], 0)
        self.assertFalse(out.loc["E", "indirect_alert_in"])

    def test_float_ids_from_csv_with_nan_match_int_ids(self):
        # Un CSV con NaN convierte la columna a float: 1 -> 1.0
        direct = pd.DataFrame({"id": [1, 2], "direct_alert": [True, False]})
        mov = pd.DataFrame({"origen_id": [1.0, np.nan], "destination_id": [2.0, 2.0]})
        out = self.by_id(run(direct, mov))
        self.assertTrue(out.loc["2", "indirect_alert_in"])
        self.assertEqual(out.loc["2", "n_in"], 2)
        self.assertEqual(out.loc["2", "n_indirect_in"], 1)

    def test_leading_zeros_match(self):
        direct = pd.DataFrame({"id": ["0001", "0002"], "direct_alert": [True, False]})
        mov = pd.DataFrame({"origen_id": [1], "destination_id": [2]})
        out = self.by_id(run(direct, mov))
        # La salida conserva el ID original
        self.assertTrue(out.loc["0002", "indirect_alert_in"])

    def test_normalize_ids_false_keeps_strict_matching(self):
        direct = pd.DataFrame({"id": [1, 2], "direct_alert": [True, False]})
        mov = pd.DataFrame({"origen_id": [1.0], "destination_id": [2.0]})
        out = self.by_id(run(direct, mov, normalize_ids=False))
        self.assertFalse(out.loc["2", "indirect_alert_in"])

    def test_nan_direct_alert_is_not_an_alert(self):
        direct = pd.DataFrame({"id": ["A", "B"], "direct_alert": [np.nan, False]})
        mov = pd.DataFrame({"origen_id": ["A"], "destination_id": ["B"]})
        out = self.by_id(run(direct, mov))
        self.assertFalse(out.loc["B", "indirect_alert_in"])

    def test_duplicate_ids(self):
        direct = pd.DataFrame({"id": ["A", "A", "B"], "direct_alert": [False, True, False]})
        mov = pd.DataFrame({"origen_id": ["A"], "destination_id": ["B"]})
        with self.assertLogs(level="WARNING"):
            out = run(direct, mov)
        self.assertEqual(out["id"].tolist(), ["A", "B"])
        self.assertTrue(self.by_id(out).loc["B", "indirect_alert_in"])

    def test_missing_movement_ids_do_not_match_anything(self):
        direct = pd.DataFrame({"id": ["A", "B"], "direct_alert": [True, False]})
        mov = pd.DataFrame({"origen_id": [None, "A"], "destination_id": [np.nan, "B"]})
        out = self.by_id(run(direct, mov))
        self.assertEqual(out.loc["A", "n_out"], 1)
        self.assertEqual(out.loc["B", "n_in"], 1)

    def test_no_overlap_warns(self):
        mov = pd.DataFrame({"origen_id": ["X"], "destination_id": ["Y"]})
        with self.assertLogs(level="WARNING"):
            out = run(self.direct, mov)
        self.assertFalse(out["indirect_alert_in"].any())
        self.assertFalse(out["indirect_alert_out"].any())

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            run(self.direct, pd.DataFrame({"origen_id": ["A"]}))
        with self.assertRaises(ValueError):
            run(pd.DataFrame({"id": ["A"]}), self.mov)


if __name__ == "__main__":
    unittest.main()
