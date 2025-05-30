import unittest
import pandas as pd
from ganabosques_risk_package.risk_level import RiskLevel
from ganabosques_risk_package.calculate_risk_administrative import calculate_risk_administrative

class TestCalculateRiskAdministrative(unittest.TestCase):

    def setUp(self):
        # Simulated plot data
        self.df_plots = pd.DataFrame({
            "plot_id": [1, 2],
            "risk_total": [RiskLevel.HIGH, RiskLevel.LOW],
            "adm1_id": [101, 101],
            "adm1_name": ["DeptA", "DeptA"],
            "adm2_id": [201, 202],
            "adm2_name": ["MunA", "MunB"],
            "adm3_id": [301, 302],
            "adm3_name": ["VerA", "VerB"]
        })

        # Simulated enterprise data
        self.df_enterprises = pd.DataFrame({
            "enterprise_id": [10],
            "enterprise_risk_enum": [RiskLevel.MEDIUM],
            "adm1_id": [101],
            "adm1_name": ["DeptA"],
            "adm2_id": [201],
            "adm2_name": ["MunA"],
            "adm3_id": [301],
            "adm3_name": ["VerA"]
        })

    def test_output_structure(self):
        # Test: validate that output contains the expected columns
        result = calculate_risk_administrative(self.df_plots, self.df_enterprises)
        expected_columns = {
            "adm1_id", "adm1_name", "adm1_score", "adm1_risk",
            "adm2_id", "adm2_name", "adm2_score", "adm2_risk",
            "adm3_id", "adm3_name", "adm3_score", "adm3_risk"
        }
        self.assertTrue(expected_columns.issubset(result.columns))

    def test_non_empty_output(self):
        # Test: validate output is not empty
        result = calculate_risk_administrative(self.df_plots, self.df_enterprises)
        self.assertGreater(len(result), 0)

    def test_risk_enum_type(self):
        # Test: validate that risk levels are enums
        result = calculate_risk_administrative(self.df_plots, self.df_enterprises)
        self.assertTrue(isinstance(result.iloc[0]["adm1_risk"], RiskLevel))
        self.assertTrue(isinstance(result.iloc[0]["adm2_risk"], RiskLevel))
        self.assertTrue(isinstance(result.iloc[0]["adm3_risk"], RiskLevel))

if __name__ == "__main__":
    unittest.main()
