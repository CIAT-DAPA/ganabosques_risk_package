import unittest
from ganabosques_risk_package.risk_level import RiskLevel

class TestRiskLevelEnum(unittest.TestCase):
    def test_enum_members_exist(self):
        # Test: RiskLevel has all expected members
        self.assertTrue(hasattr(RiskLevel, "LOW"))
        self.assertTrue(hasattr(RiskLevel, "MEDIUM"))
        self.assertTrue(hasattr(RiskLevel, "HIGH"))
        self.assertTrue(hasattr(RiskLevel, "NO_RISK"))

    def test_enum_values_are_correct(self):
        # Test: RiskLevel values match expected Spanish labels
        self.assertEqual(RiskLevel.LOW.value, 1)
        self.assertEqual(RiskLevel.MEDIUM.value, 2)
        self.assertEqual(RiskLevel.HIGH.value, 3)
        self.assertEqual(RiskLevel.NO_RISK.value, 0)

    def test_enum_names(self):
        # Test: Enum names are in uppercase
        self.assertEqual(RiskLevel.LOW.name, "LOW")
        self.assertEqual(RiskLevel.MEDIUM.name, "MEDIUM")
        self.assertEqual(RiskLevel.HIGH.name, "HIGH")
        self.assertEqual(RiskLevel.NO_RISK.name, "NO_RISK")

    def test_enum_type(self):
        # Test: RiskLevel members are instances of RiskLevel
        for level in RiskLevel:
            self.assertIsInstance(level, RiskLevel)

if __name__ == "__main__":
    unittest.main()