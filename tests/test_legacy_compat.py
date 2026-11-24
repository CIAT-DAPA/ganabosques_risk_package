"""Compatibilidad entre la API vieja (plot_alert_*, entity_alert) y la nueva."""

import contextlib
import io
import unittest
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ganabosques_risk_package import alert_indirect, total_risk
from ganabosques_risk_package.entity_alert import calculate_alert
from ganabosques_risk_package.plot_alert_direct import alert_direct as legacy_alert_direct
from ganabosques_risk_package.plot_alert_direct_serial import alert_direct_serial
from ganabosques_risk_package.plot_alert_indirect import alert_indirect as legacy_alert_indirect

DATA = Path(__file__).resolve().parent / "data_test"


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


class TestLegacyDeprecation(unittest.TestCase):

    def test_importing_legacy_modules_does_not_silence_warnings(self):
        # Antes, importar plot_alert_indirect/entity_alert hacía
        # warnings.filterwarnings("ignore") para todo el proceso.
        self.assertFalse(any(
            f[0] == "ignore" and f[2] is Warning and f[1] is None
            for f in warnings.filters
        ))

    def test_legacy_indirect_warns(self):
        ad = pd.DataFrame({"id": ["A", "B"], "alert_direct": [True, False]})
        mv = pd.DataFrame({"origen_id": ["A"], "destination_id": ["B"]})
        with self.assertWarns(DeprecationWarning) as cm:
            quiet(legacy_alert_indirect, ad, mv, n_workers=1)
        self.assertIn("indirect_alert_in", str(cm.warning))
        self.assertTrue(cm.filename.endswith("test_legacy_compat.py"))

    def test_legacy_direct_warns(self):
        plots = gpd.read_file(DATA / "plot_not_intersect" / "plot_not_intersect.shp")
        areas = str(DATA / "areas" / "areas.shp")
        raster = str(DATA / "deforestation.tif")
        for fn in (legacy_alert_direct, alert_direct_serial):
            with self.subTest(fn=fn.__name__), self.assertWarns(DeprecationWarning):
                quiet(fn, plots, raster, areas, areas, n_workers=1)


class TestEntityAlertWithNewApi(unittest.TestCase):

    def setUp(self):
        self.entity = pd.DataFrame({"id": ["X", "Y", "Z"], "name": ["Alpha", "Beta", "Gamma"]})
        self.provider = pd.DataFrame({
            "plot_id": [1, 2, 3, 4],
            "entity_id": ["X", "X", "Y", "Z"],
        })

    def test_new_api_output_matches_legacy_semantics(self):
        # 1 tiene alerta directa; movimientos 1->2 y 3->1
        direct = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "direct_alert": [True, False, False, False],
            "deforested_ha": [2.5, 0.0, 0.0, 0.0],
            "deforested_prop": [0.1, 0.0, 0.0, 0.0],
        })
        mov = pd.DataFrame({"origen_id": [1, 3], "destination_id": [2, 1]})
        risk = quiet(total_risk, direct, quiet(alert_indirect, direct, mov))
        out = quiet(calculate_alert, risk, self.entity, self.provider, n_workers=1).set_index("entity_id")

        # Mismo cálculo con la API vieja
        legacy_in = direct.rename(columns={"direct_alert": "alert_direct", "deforested_ha": "deforested_area"})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy_ind = quiet(legacy_alert_indirect, legacy_in, mov, n_workers=1)
        legacy = quiet(calculate_alert, legacy_ind, self.entity, self.provider, n_workers=1).set_index("entity_id")

        pd.testing.assert_frame_equal(out, legacy)
        # X: plot 1 (directa) y plot 2 (recibe de 1 -> alert_out viejo)
        self.assertEqual(out.loc["X", "plots_alert_direct"], 1)
        self.assertEqual(out.loc["X", "plots_alert_out"], 1)
        self.assertEqual(out.loc["X", "deforested_area_sum"], 2.5)
        # Y: plot 3 envía a 1 (con alerta) -> alert_in viejo
        self.assertEqual(out.loc["Y", "plots_alert_in"], 1)
        self.assertFalse(out.loc["Z", "alert"])

    def test_no_info_and_nan_are_not_alerts(self):
        # total_risk sin indirect_df llena las columnas con "no_info"
        direct = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "direct_alert": [False, None, False, False],
            "deforested_ha": [0.0, 0.0, 0.0, 0.0],
        })
        risk = quiet(total_risk, direct)
        out = quiet(calculate_alert, risk, self.entity, self.provider, n_workers=1)
        self.assertFalse(out["alert"].any())

    def test_int_and_str_plot_ids_match(self):
        plots = pd.DataFrame({
            "id": ["0001", "2"],
            "deforested_area": [1.0, 0.0],
            "alert_direct": [True, False],
            "alert_in": [False, False],
            "alert_out": [False, False],
        })
        out = quiet(calculate_alert, plots, self.entity, self.provider, n_workers=1).set_index("entity_id")
        self.assertEqual(out.loc["X", "plots_alert_direct"], 1)
        self.assertEqual(out.loc["X", "deforested_area_sum"], 1.0)


if __name__ == "__main__":
    unittest.main()
