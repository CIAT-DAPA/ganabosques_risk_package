import unittest
import pandas as pd
from ganabosques_risk_package.calculate_risk_movement import calculate_risk_movement

class TestCalculateRiskMovement(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame([
            {"id_origen": 1, "id_destino": 2, "tipo": "salida", "nivel_riesgo": "BAJO"},
            {"id_origen": 2, "id_destino": 1, "tipo": "entrada", "nivel_riesgo": "ALTO"},
            {"id_origen": 3, "id_destino": 1, "tipo": "entrada", "nivel_riesgo": "MEDIO"},
            {"id_origen": 1, "id_destino": 4, "tipo": "salida", "nivel_riesgo": "MEDIO"},
        ])

    def test_movement_risks(self):
        # Comentario: prueba que el cálculo de riesgo promedio de entrada y salida sea correcto
        result = calculate_risk_movement(1, self.data)
        self.assertEqual(result['riesgo_entrada_promedio'], 'ALTO')   # Promedio (3 + 2) / 2 = 2.5
        self.assertEqual(result['riesgo_salida_promedio'], 'MEDIO')  # Promedio (1 + 2) / 2 = 1.5

    def test_no_movements(self):
        # Comentario: prueba que se retorne BAJO si no hay datos de entrada ni salida
        result = calculate_risk_movement(5, self.data)
        self.assertEqual(result['riesgo_entrada_promedio'], 'BAJO')
        self.assertEqual(result['riesgo_salida_promedio'], 'BAJO')

if __name__ == "__main__":
    unittest.main()