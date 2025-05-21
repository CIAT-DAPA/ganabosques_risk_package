import unittest
import pandas as pd
from ganabosques_risk_package.calculate_risk_total import calculate_risk_total

class TestCalculateRiskTotal(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame([
            {"id": 1, "risk_direct_level": "ALTO", "riesgo_entrada_promedio": "ALTO", "riesgo_salida_promedio": "ALTO"},
            {"id": 2, "risk_direct_level": "BAJO", "riesgo_entrada_promedio": "BAJO", "riesgo_salida_promedio": "BAJO"},
            {"id": 3, "risk_direct_level": "MEDIO", "riesgo_entrada_promedio": "MEDIO", "riesgo_salida_promedio": "MEDIO"},
            {"id": 4, "risk_direct_level": "BAJO", "riesgo_entrada_promedio": "ALTO", "riesgo_salida_promedio": "MEDIO"},
        ])

    def test_total_risk_classification(self):
        # Comentario: prueba que los riesgos totales se clasifiquen correctamente según los promedios
        result = calculate_risk_total(self.df)
        self.assertEqual(result[result['id'] == 1]['riesgo_total'].values[0], 'ALTO')   # Prom 3.0
        self.assertEqual(result[result['id'] == 2]['riesgo_total'].values[0], 'BAJO')   # Prom 1.0
        self.assertEqual(result[result['id'] == 3]['riesgo_total'].values[0], 'MEDIO')  # Prom 2.0
        self.assertEqual(result[result['id'] == 4]['riesgo_total'].values[0], 'MEDIO')  # Prom 2.0

if __name__ == "__main__":
    unittest.main()