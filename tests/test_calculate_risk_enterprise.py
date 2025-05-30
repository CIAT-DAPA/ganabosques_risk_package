import unittest
import pandas as pd
from ganabosques_risk_package.risk_level import RiskLevel
from ganabosques_risk_package.calculate_risk_enterprise import calculate_risk_enterprise

class TestCalculateRiskEnterprise(unittest.TestCase):

    def setUp(self):
        # Empresas
        self.df_enterprises = pd.DataFrame({
            "id": [1, 2]
        })

        # Relación empresa - predios
        self.df_suppliers = pd.DataFrame({
            "enterprise_id": [1, 1, 2],
            "plot_id": [10, 11, 12]
        })

        # Riesgos de predios
        self.df_plots_risk = pd.DataFrame({
            "plot_id": [10, 11, 12],
            "risk_total": [RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
        })

    def test_output_structure(self):
        # Verifica que las columnas de salida sean las esperadas
        result = calculate_risk_enterprise(self.df_enterprises, self.df_suppliers, self.df_plots_risk)
        expected_columns = {
            "enterprise_id", "enterprise_risk", "enterprise_risk_value", "enterprise_risk_label",
            "criteria_deforestation_high", "criteria_location_restricted",
            "criteria_municipality_critical", "criteria_geolocation_coverage"
        }
        self.assertTrue(expected_columns.issubset(result.columns))

    def test_risk_is_enum(self):
        # Verifica que el riesgo asignado sea del tipo RiskLevel
        result = calculate_risk_enterprise(self.df_enterprises, self.df_suppliers, self.df_plots_risk)
        self.assertTrue(isinstance(result.loc[0, "enterprise_risk"], RiskLevel))

    def test_coverage_threshold(self):
        # Verifica que el criterio de cobertura geoespacial se evalúe correctamente
        partial_suppliers = pd.DataFrame({
            "enterprise_id": [1, 1, 2],
            "plot_id": [10, 13, 14]  # 13 y 14 no están geolocalizados
        })
        result = calculate_risk_enterprise(self.df_enterprises, partial_suppliers, self.df_plots_risk)
        self.assertFalse(result[result["enterprise_id"] == 1]["criteria_geolocation_coverage"].values[0])

if __name__ == "__main__":
    unittest.main()
