import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from ganabosques_risk_package.alert_enterprise import alert_enterprise

OUT_COLS = [
    "idpro", "id_farm", "typemove", "enterprise_type",
    "farm_has_direct_alert", "farm_has_indirect_alert",
]


def run(*args, **kwargs) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        return alert_enterprise(*args, **kwargs)


class TestAlertEnterprise(unittest.TestCase):

    def setUp(self):
        self.risk = pd.DataFrame({
            "id": ["F1", "F2", "F3"],
            "direct_alert": [True, False, False],
            "indirect_alert_in": [False, True, False],
            "indirect_alert_out": [False, False, False],
        })
        self.mov = pd.DataFrame({
            "origen_id": ["F1", "S1", "F3", "F1", "F1"],
            "destination_id": ["S1", "F2", "S1", "F2", "S1"],
            "tipo_origen": ["FARM", "SLAUGHTERHOUSE", "FARM", "FARM", "FARM"],
            "tipo_destino": ["SLAUGHTERHOUSE", "FARM", "SLAUGHTERHOUSE", "FARM", "SLAUGHTERHOUSE"],
        })

    def test_in_and_out(self):
        out = run(self.risk, self.mov)
        self.assertEqual(list(out.columns), OUT_COLS)
        rows = set(zip(out["idpro"], out["id_farm"], out["typemove"]))
        # F1 (alerta directa) envía a S1; S1 envía a F2 (alerta indirecta).
        # F3 sin alerta y el movimiento finca->finca F1->F2 no cuentan. Duplicado F1->S1 se elimina.
        self.assertEqual(rows, {("S1", "F1", "in"), ("S1", "F2", "out")})
        f1 = out[out["id_farm"] == "F1"].iloc[0]
        self.assertTrue(f1["farm_has_direct_alert"])
        self.assertFalse(f1["farm_has_indirect_alert"])
        f2 = out[out["id_farm"] == "F2"].iloc[0]
        self.assertEqual(f2["enterprise_type"], "SLAUGHTERHOUSE")
        self.assertTrue(f2["farm_has_indirect_alert"])

    def test_uppercase_columns_and_producer_ids(self):
        mov = pd.DataFrame({
            "SIT_CODE_ORIGEN": ["F1"],
            "SIT_CODE_DESTINO": ["S1"],
            "TIPO_ORIGEN": ["farm"],
            "TIPO_DESTINO": [" cattle_fair "],
            "PRODUCER_ID_DESTINO": ["0099"],
        })
        out = run(self.risk, mov)
        self.assertEqual(out["idpro"].tolist(), ["99"])
        self.assertEqual(out["enterprise_type"].tolist(), ["CATTLE_FAIR"])

    def test_numeric_ids_are_normalized(self):
        risk = pd.DataFrame({"id": [1], "direct_alert": [True]})
        mov = pd.DataFrame({
            "origen_id": [1.0], "destination_id": ["S1"],
            "tipo_origen": ["FARM"], "tipo_destino": ["ENTERPRISE"],
        })
        self.assertEqual(run(risk, mov)["id_farm"].tolist(), ["1"])

    def test_enterprise_types_filter(self):
        out = run(self.risk, self.mov, enterprise_types=["CATTLE_FAIR"])
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), OUT_COLS)

    def test_nan_and_no_info_are_not_alerts(self):
        risk = pd.DataFrame({
            "id": ["F1"],
            "direct_alert": [np.nan],
            "indirect_alert_in": ["no_info"],
        })
        with self.assertLogs(level="WARNING"):
            out = run(risk, self.mov)
        self.assertTrue(out.empty)

    def test_no_alert_columns(self):
        with self.assertLogs(level="WARNING"):
            out = run(pd.DataFrame({"id": ["F1"]}), self.mov)
        self.assertTrue(out.empty)

    def test_without_type_columns_farm_to_farm_is_not_enterprise(self):
        mov = self.mov.drop(columns=["tipo_origen", "tipo_destino"])
        with self.assertLogs(level="WARNING"):
            out = run(self.risk, mov)
        rows = set(zip(out["idpro"], out["id_farm"], out["typemove"]))
        # F1->F2 es finca->finca (F2 está en total_risk_df): no cuenta
        self.assertEqual(rows, {("S1", "F1", "in"), ("S1", "F2", "out")})
        self.assertTrue((out["enterprise_type"] == "").all())

    def test_missing_movement_columns_raise(self):
        with self.assertRaises(ValueError):
            run(self.risk, pd.DataFrame({"origen_id": ["F1"]}))


if __name__ == "__main__":
    unittest.main()
