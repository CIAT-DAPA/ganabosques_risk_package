import contextlib
import io
import unittest

import numpy as np
import pandas as pd

from ganabosques_risk_package.supplier_risk import (
    filter_suppliers_by_period,
    get_years_for_period,
    supplier_risk,
    supplier_risk_summary,
)


def run(*args, **kwargs) -> pd.DataFrame:
    with contextlib.redirect_stdout(io.StringIO()):
        return supplier_risk(*args, **kwargs)


class TestPeriods(unittest.TestCase):

    def test_get_years_for_period(self):
        self.assertEqual(get_years_for_period("2017", "annual"), {2017})
        self.assertEqual(get_years_for_period("2015-2017", "cumulative"), {2015, 2016, 2017})
        self.assertEqual(get_years_for_period("2015-2016", "annual"), {2015, 2016})
        self.assertEqual(get_years_for_period("201703", "nad"), {2017})
        self.assertEqual(get_years_for_period("201703", "atd"), {2017})
        self.assertEqual(get_years_for_period("abc", "annual"), set())

    def test_filter_disabled_returns_all(self):
        s = pd.DataFrame({"enterprise_id": ["E1", "E2"], "farm_id": ["A", "B"], "years": [[2010], [2020]]})
        self.assertEqual(len(filter_suppliers_by_period(s, "2017", "annual")), 2)

    def test_filter_years_formats(self):
        s = pd.DataFrame({
            "enterprise_id": ["E1", "E2", "E3", "E4", "E5"],
            "farm_id": ["A", "B", "C", "D", "E"],
            "years": [[2016, 2017], "2017, 2018", 2017, [2010], None],
        })
        out = filter_suppliers_by_period(s, "2017", "annual", filter_by_year=True)
        self.assertEqual(out["enterprise_id"].tolist(), ["E1", "E2", "E3"])

    def test_filter_years_numpy_arrays_and_strings(self):
        # Listas leídas de parquet/Mongo llegan como numpy arrays
        s = pd.DataFrame({
            "enterprise_id": ["E1", "E2", "E3", "E4"],
            "farm_id": ["A", "B", "C", "D"],
            "years": [np.array([2016, 2017]), np.array([2010]), ["2017"], ["abc", 2017.0]],
        })
        out = filter_suppliers_by_period(s, "2017", "annual", filter_by_year=True)
        self.assertEqual(out["enterprise_id"].tolist(), ["E1", "E3", "E4"])


class TestSupplierRisk(unittest.TestCase):

    def setUp(self):
        self.risk = pd.DataFrame({
            "id": ["1", "2", "3", "4"],
            "direct_alert": [True, False, False, False],
            "indirect_alert_in": [False, True, False, False],
            "indirect_alert_out": [False, False, True, False],
            "deforested_ha": [2.0, 0.0, 0.0, 0.0],
            "deforested_prop": [0.2, 0.0, 0.0, 0.0],
        })
        self.sups = pd.DataFrame({
            "enterprise_id": ["E1", "E1", "E2", "E3", "E4"],
            "farm_id": [1, "0002", 3.0, "4", "99"],
            "enterprise_name": ["Uno", "Uno", "Dos", "Tres", "Cuatro"],
        })

    def test_basic(self):
        out = run(self.risk, self.sups)
        self.assertEqual(sorted(zip(out["enterprise_id"], out["farm_id"])),
                         [("E1", "1"), ("E1", "2"), ("E2", "3")])
        # Se preservan columnas extra de suppliers
        self.assertIn("enterprise_name", out.columns)
        row = out.set_index("farm_id").loc["1"]
        self.assertTrue(row["risk_direct"])
        self.assertEqual(row["deforested_ha"], 2.0)

    def test_nan_alerts_are_not_risk(self):
        # Huecos de un outer merge previo (NaN) no deben interpretarse como alerta
        risk = self.risk.assign(
            direct_alert=[np.nan] * 4,
            indirect_alert_in=[np.nan] * 4,
            indirect_alert_out=["no_info"] * 4,
        )
        out = run(risk, self.sups)
        self.assertTrue(out.empty)

    def test_farms_without_risk_info_are_excluded(self):
        out = run(self.risk, self.sups)
        self.assertNotIn("99", out["farm_id"].tolist())

    def test_filter_by_year(self):
        sups = self.sups.assign(years=[[2017], [2010], [2017], [2017], [2017]])
        out = run(self.risk, sups, period="2017", period_type="annual", filter_by_year=True)
        self.assertEqual(sorted(out["farm_id"]), ["1", "3"])

    def test_missing_columns_raise(self):
        with self.assertRaises(ValueError):
            run(self.risk, self.sups.drop(columns="farm_id"))
        with self.assertRaises(ValueError):
            run(self.risk, self.sups.drop(columns="enterprise_id"))
        with self.assertRaises(ValueError):
            run(self.risk.drop(columns="id"), self.sups)

    def test_summary(self):
        summary = supplier_risk_summary(run(self.risk, self.sups)).set_index("enterprise_id")
        self.assertEqual(summary.loc["E1", "n_farms_at_risk"], 2)
        self.assertEqual(summary.loc["E1", "n_risk_direct"], 1)
        self.assertEqual(summary.loc["E1", "total_deforested_ha"], 2.0)
        self.assertEqual(summary.index[0], "E1")  # ordenado por n_farms_at_risk desc

    def test_summary_empty(self):
        self.assertTrue(supplier_risk_summary(pd.DataFrame()).empty)

    def test_summary_without_deforested_ha(self):
        df = run(self.risk.drop(columns=["deforested_ha", "deforested_prop"]), self.sups)
        summary = supplier_risk_summary(df).set_index("enterprise_id")
        self.assertEqual(summary.loc["E1", "total_deforested_ha"], 0.0)
        self.assertEqual(summary.loc["E1", "n_farms_at_risk"], 2)

    def test_summary_counts_unique_farms(self):
        # Una fila por año: la misma relación E1-1 aparece 3 veces
        sups = pd.DataFrame({
            "enterprise_id": ["E1", "E1", "E1", "E1"],
            "farm_id": ["1", "1", "1", "2"],
            "year": [2015, 2016, 2017, 2017],
        })
        summary = supplier_risk_summary(run(self.risk, sups)).set_index("enterprise_id")
        self.assertEqual(summary.loc["E1", "n_farms_at_risk"], 2)
        self.assertEqual(summary.loc["E1", "n_risk_direct"], 1)
        self.assertEqual(summary.loc["E1", "total_deforested_ha"], 2.0)  # no 6.0


if __name__ == "__main__":
    unittest.main()
