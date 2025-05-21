import unittest
import pandas as pd
from ganabosques_risk_package.calculate_risk_administrative import calculate_risk_administrative

class TestCalculateRiskAdministrative(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"vereda": "V1", "riesgo_total": "ALTO"},
            {"vereda": "V1", "riesgo_total": "ALTO"},
            {"vereda": "V1", "riesgo_total": "MEDIO"},
            {"vereda": "V2", "riesgo_total": "MEDIO"},
            {"vereda": "V2", "riesgo_total": "MEDIO"},
            {"vereda": "V2", "riesgo_total": "BAJO"},
            {"vereda": "V3", "riesgo_total": "BAJO"},
            {"vereda": "V3", "riesgo_total": "MEDIO"},
            {"vereda": "V3", "riesgo_total": "BAJO"},
        ])

    def test_vereda_risk_levels(self):
        # Comentario: verifica que se clasifiquen correctamente las veredas por nivel de riesgo
        result = calculate_risk_administrative(self.df)
        v1 = result[result['vereda'] == "V1"].iloc[0]
        v2 = result[result['vereda'] == "V2"].iloc[0]
        v3 = result[result['vereda'] == "V3"].iloc[0]

        self.assertEqual(v1['riesgo_vereda'], 'ALTO')
        self.assertEqual(v2['riesgo_vereda'], 'MEDIO')
        self.assertEqual(v3['riesgo_vereda'], 'BAJO')

if __name__ == "__main__":
    unittest.main()