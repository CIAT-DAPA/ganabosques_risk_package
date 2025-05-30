import unittest
import pandas as pd
from ganabosques_risk_package.risk_level import RiskLevel
from ganabosques_risk_package.calculate_risk_total import calculate_risk_total

class TestCalculateRiskTotal(unittest.TestCase):

    def setUp(self):
        # Simulated direct risk results
        self.df_direct = pd.DataFrame({
            'id': [1, 2, 3],
            'risk_direct_level': [RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
        })

        # Simulated movement risk results
        self.df_movement = pd.DataFrame({
            'plot_id': [1, 2, 3],
            'risk_in_score': [2.0, 1.0, 0.0],
            'risk_out_score': [1.0, 2.0, 0.0]
        })

    def test_output_structure(self):
        # Test: check if all columns exist in the result
        result = calculate_risk_total(self.df_direct, self.df_movement)
        expected_columns = {
            "plot_id", "risk_direct_value", "risk_in_score", "risk_out_score",
            "risk_total_score", "risk_total", "risk_total_value", "risk_total_label"
        }
        self.assertTrue(expected_columns.issubset(result.columns))
        self.assertEqual(len(result), 3)

    def test_enum_and_score_relationship(self):
        # Test: consistency between total score and enum classification
        result = calculate_risk_total(self.df_direct, self.df_movement)
        for _, row in result.iterrows():
            self.assertEqual(row['risk_total_value'], row['risk_total'].value)
            self.assertEqual(row['risk_total_label'], row['risk_total'].name)

    def test_missing_movement_defaults(self):
        # Test: one plot without movement data defaults to zero risk_in and risk_out
        df_direct_partial = pd.DataFrame({
            'id': [99],
            'risk_direct_level': [RiskLevel.LOW]
        })
        result = calculate_risk_total(df_direct_partial, self.df_movement)
        self.assertEqual(result.iloc[0]['risk_in_score'], 0)
        self.assertEqual(result.iloc[0]['risk_out_score'], 0)

if __name__ == "__main__":
    unittest.main()
