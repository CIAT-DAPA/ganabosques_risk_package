"""Tests de spatial_metrics, alert_direct (API nueva) y del pipeline completo
usando los fixtures reales de tests/data_test."""

import contextlib
import io
import unittest
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ganabosques_risk_package import (
    alert_direct,
    alert_enterprise,
    alert_indirect,
    spatial_metrics,
    supplier_risk,
    total_risk,
)

DATA = Path(__file__).resolve().parent / "data_test"
RASTER = str(DATA / "deforestation.tif")
PIXEL_HA = 30 * 30 / 10_000  # raster EPSG:3116 a 30 m


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, show_progress=False, **kwargs)


class TestSpatialPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        inside = gpd.read_file(DATA / "plot_intersect" / "plot_intersect.shp").assign(id="P_IN")
        outside = gpd.read_file(DATA / "plot_not_intersect" / "plot_not_intersect.shp").assign(id="P_OUT")
        # Plots en EPSG:4326: se reproyectan internamente a EPSG:3116
        cls.plots = gpd.GeoDataFrame(
            pd.concat([inside, outside], ignore_index=True)[["id", "geometry"]],
            crs=inside.crs,
        )
        cls.areas = gpd.read_file(DATA / "areas" / "areas.shp")
        cls.metrics = quiet(spatial_metrics, cls.plots, cls.areas, cls.areas)

    # ------------------------------------------------------------------ metrics

    def test_metrics_values(self):
        m = self.metrics.set_index("id")
        self.assertEqual(set(m.index), {"P_IN", "P_OUT"})
        self.assertGreater(m.loc["P_IN", "protected_ha"], 0)
        self.assertGreater(m.loc["P_IN", "farming_in_ha"], 0)
        self.assertEqual(m.loc["P_OUT", "protected_ha"], 0)
        self.assertEqual(m.loc["P_OUT", "farming_out_prop"], 1.0)
        for pid in m.index:
            self.assertAlmostEqual(
                m.loc[pid, "farming_in_ha"] + m.loc[pid, "farming_out_ha"],
                m.loc[pid, "total_ha"], places=2,
            )
        for c in ["farming_in_prop", "farming_out_prop", "protected_prop"]:
            self.assertTrue(m[c].between(0, 1).all(), c)

    def test_metrics_without_reference_layers(self):
        with self.assertLogs(level="WARNING"):
            m = quiet(spatial_metrics, self.plots)
        self.assertTrue((m["protected_ha"] == 0).all())
        self.assertTrue((m["total_ha"] > 0).all())

    def test_metrics_missing_id_column(self):
        with self.assertRaises(ValueError):
            quiet(spatial_metrics, self.plots.rename(columns={"id": "x"}))

    # ------------------------------------------------------------------ direct

    def _check_direct(self, out):
        d = out.set_index("id")
        self.assertTrue(d.loc["P_IN", "direct_alert"])
        self.assertGreater(d.loc["P_IN", "deforested_ha"], 0)
        self.assertTrue(0 < d.loc["P_IN", "deforested_prop"] <= 1)
        self.assertFalse(d.loc["P_OUT", "direct_alert"])
        self.assertEqual(d.loc["P_OUT", "deforested_ha"], 0)
        return d

    def test_alert_direct_fast(self):
        d = self._check_direct(quiet(alert_direct, self.plots, RASTER))
        # Modo rápido: el área es un múltiplo exacto del área del píxel
        n_pixels = d.loc["P_IN", "deforested_ha"] / PIXEL_HA
        self.assertAlmostEqual(n_pixels, round(n_pixels), places=3)

    def test_alert_direct_precise(self):
        self._check_direct(quiet(alert_direct, self.plots, RASTER, use_precise_area=True))

    def test_alert_direct_with_metrics(self):
        out = quiet(alert_direct, self.plots, RASTER, metrics_df=self.metrics)
        self.assertIn("protected_ha", out.columns)
        self.assertEqual(len(out), 2)

    def test_alert_direct_missing_raster(self):
        with self.assertRaises(FileNotFoundError):
            quiet(alert_direct, self.plots, str(DATA / "no_existe.tif"))

    # ------------------------------------------------------------------ pipeline

    def test_end_to_end(self):
        direct = quiet(alert_direct, self.plots, RASTER, metrics_df=self.metrics)
        movements = pd.DataFrame({
            "origen_id": ["P_IN", "S1", "P_OUT"],
            "destination_id": ["P_OUT", "P_OUT", "S1"],
            "tipo_origen": ["FARM", "SLAUGHTERHOUSE", "FARM"],
            "tipo_destino": ["FARM", "FARM", "SLAUGHTERHOUSE"],
        })
        indirect = quiet(alert_indirect, direct, movements)
        risk = quiet(total_risk, direct, indirect).set_index("id")

        self.assertTrue(risk.loc["P_IN", "direct_alert"])
        self.assertFalse(risk.loc["P_OUT", "direct_alert"])
        self.assertTrue(risk.loc["P_OUT", "indirect_alert_in"])   # recibe de P_IN
        self.assertFalse(risk.loc["P_IN", "indirect_alert_out"])  # envía a P_OUT, sin alerta directa
        self.assertGreater(risk.loc["P_OUT", "total_ha"], 0)

        ent = quiet(alert_enterprise, risk.reset_index(), movements)
        self.assertEqual(set(zip(ent["idpro"], ent["id_farm"], ent["typemove"])),
                         {("S1", "P_OUT", "in"), ("S1", "P_OUT", "out")})

        sups = pd.DataFrame({"enterprise_id": ["E1", "E2"], "farm_id": ["P_IN", "NO_EXISTE"]})
        sr = quiet(supplier_risk, risk.reset_index(), sups)
        self.assertEqual(sr["enterprise_id"].tolist(), ["E1"])


if __name__ == "__main__":
    unittest.main()
