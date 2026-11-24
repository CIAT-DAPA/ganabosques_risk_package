import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from ganabosques_risk_package.total_risk import total_risk


def run(*args, **kwargs) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        return total_risk(*args, **kwargs)


def make_indirect(ids, alert_in=None, alert_out=None):
    n = len(ids)
    alert_in = alert_in or [False] * n
    alert_out = alert_out or [False] * n
    return pd.DataFrame({
        "id": ids,
        "n_in": [int(a) for a in alert_in],
        "n_out": [int(a) for a in alert_out],
        "n_indirect_in": [int(a) for a in alert_in],
        "n_indirect_out": [int(a) for a in alert_out],
        "n_total_mov": [int(a) + int(b) for a, b in zip(alert_in, alert_out)],
        "indirect_alert_in": alert_in,
        "indirect_alert_out": alert_out,
    })


class TestTotalRisk(unittest.TestCase):

    def setUp(self):
        self.direct = pd.DataFrame({
            "id": [1, 2, 3],
            "direct_alert": [True, False, False],
            "deforested_ha": [1.5, 0.0, 0.0],
            "deforested_prop": [0.1, 0.0, 0.0],
        })
        self.metrics = pd.DataFrame({
            "id": [1, 2, 3],
            "total_ha": [15.0, 10.0, 5.0],
            "farming_in_ha": [15.0, 5.0, 0.0],
            "farming_in_prop": [1.0, 0.5, 0.0],
            "farming_out_ha": [0.0, 5.0, 5.0],
            "farming_out_prop": [0.0, 0.5, 1.0],
            "protected_ha": [0.0, 0.0, 1.0],
            "protected_prop": [0.0, 0.0, 0.2],
        })

    def test_full_merge(self):
        ind = make_indirect(["1", "2", "3"], alert_in=[False, True, False])
        out = run(self.direct, ind, self.metrics).set_index("id")
        self.assertEqual(len(out), 3)
        self.assertTrue(out.loc["1", "direct_alert"])
        self.assertTrue(out.loc["2", "indirect_alert_in"])
        self.assertEqual(out.loc["3", "protected_ha"], 1.0)
        self.assertEqual(out["direct_alert"].dtype, bool)

    def test_mismatched_ids_do_not_create_false_alerts(self):
        # direct e indirect de corridas distintas: "3" falta en indirect, "9" solo existe en indirect
        ind = make_indirect(["1", "2", "9"])
        with self.assertLogs(level="WARNING"):
            out = run(self.direct, ind).set_index("id")
        self.assertEqual(set(out.index), {"1", "2", "3", "9"})
        self.assertFalse(out.loc["3", "indirect_alert_in"])
        self.assertFalse(out.loc["3", "indirect_alert_out"])
        self.assertFalse(out.loc["9", "direct_alert"])
        self.assertEqual(out.loc["9", "deforested_ha"], 0.0)
        for c in ["direct_alert", "indirect_alert_in", "indirect_alert_out"]:
            self.assertEqual(out[c].dtype, bool, c)
            self.assertFalse(out[c].isna().any(), c)
        for c in ["n_in", "n_out", "n_indirect_in", "n_indirect_out", "n_total_mov"]:
            self.assertFalse(out[c].isna().any(), c)

    def test_nan_direct_alert_is_false(self):
        direct = self.direct.assign(direct_alert=[np.nan, "True", "no_info"])
        out = run(direct).set_index("id")
        self.assertEqual(out["direct_alert"].tolist(), [False, True, False])

    def test_without_indirect_fills_no_info(self):
        out = run(self.direct)
        self.assertTrue((out["indirect_alert_in"] == "no_info").all())
        self.assertTrue((out["n_total_mov"] == "no_info").all())

    def test_metrics_default_to_zero_when_missing(self):
        out = run(self.direct)
        self.assertTrue((out["total_ha"] == 0.0).all())

    def test_metrics_not_duplicated_if_already_in_direct(self):
        direct = self.direct.merge(self.metrics, on="id")
        out = run(direct, None, self.metrics)
        self.assertNotIn("total_ha_x", out.columns)
        self.assertNotIn("total_ha_y", out.columns)

    def test_missing_required_columns_raise(self):
        with self.assertRaises(ValueError):
            run(self.direct.drop(columns="direct_alert"))
        with self.assertRaises(ValueError):
            run(self.direct.drop(columns="id"))


if __name__ == "__main__":
    unittest.main()
