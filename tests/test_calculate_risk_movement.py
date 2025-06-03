import unittest
import pandas as pd
from ganabosques_risk_package.risk_level import RiskLevel
from ganabosques_risk_package.calculate_risk_movement import calculate_risk_movement

class TestCalculateRiskMovement(unittest.TestCase):

    def setUp(self):
        # Create dummy data for plots with risk levels
        self.df_plots_risk = pd.DataFrame({
            'id': [1, 2, 3],
            'risk_direct_level': [RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
        })

        # Create dummy data for movement
        self.df_movement = pd.DataFrame({
            'origen_id': [1, 2, 2, 3],
            'destination_id': [2, 3, 1, 1]
        })

    def test_structure_of_result(self):
        # Test: verify structure and types of the result
        result = calculate_risk_movement(self.df_plots_risk, self.df_movement)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("plot_id", result.columns)
        self.assertIn("risk_in_score", result.columns)
        self.assertIn("risk_out_score", result.columns)
        self.assertIn("risk_in_enum", result.columns)
        self.assertIn("risk_out_enum_label", result.columns)
        self.assertEqual(len(result), 3)

    def test_risk_enum_output(self):
        # Test: ensure risk output is of type RiskLevel
        result = calculate_risk_movement(self.df_plots_risk, self.df_movement)
        for _, row in result.iterrows():
            self.assertTrue(isinstance(row["risk_in_enum"], RiskLevel))
            self.assertTrue(isinstance(row["risk_out_enum"], RiskLevel))

    def test_risk_score_values(self):
        # Test: validate that average scores are between 0 and 3
        result = calculate_risk_movement(self.df_plots_risk, self.df_movement)
        for _, row in result.iterrows():
            self.assertGreaterEqual(row["risk_in_score"], 0)
            self.assertLessEqual(row["risk_in_score"], 3)
            self.assertGreaterEqual(row["risk_out_score"], 0)
            self.assertLessEqual(row["risk_out_score"], 3)

    def test_empty_movements(self):
        # Test: case with no movements
        df_empty_movement = pd.DataFrame(columns=['origen_id', 'destination_id'])
        result = calculate_risk_movement(self.df_plots_risk, df_empty_movement)
        self.assertTrue((result["risk_in_enum"] == RiskLevel.NO_RISK).all())
        self.assertTrue((result["risk_out_enum"] == RiskLevel.NO_RISK).all())

if __name__ == "__main__":
    unittest.main()
