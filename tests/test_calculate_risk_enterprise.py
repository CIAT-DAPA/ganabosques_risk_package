import unittest
import pandas as pd
from ganabosques_risk_package.calculate_risk_enterprise import calculate_risk_enterprise

class TestCalculateRiskEnterprise(unittest.TestCase):
    def setUp(self):
        self.empresas_df = pd.DataFrame([
            {"empresa_id": "E1"},
            {"empresa_id": "E2"},
            {"empresa_id": "E3"},
            {"empresa_id": "E4"}
        ])
        self.proveedores_df = pd.DataFrame([
            {"empresa_id": "E1", "predio_id": 1, "riesgo_total": "ALTO"},
            {"empresa_id": "E1", "predio_id": 2, "riesgo_total": "ALTO"},
            {"empresa_id": "E1", "predio_id": 3, "riesgo_total": "BAJO"},
            {"empresa_id": "E2", "predio_id": 4, "riesgo_total": "BAJO"},
            {"empresa_id": "E2", "predio_id": 5, "riesgo_total": "BAJO"},
            {"empresa_id": "E3", "predio_id": 6, "riesgo_total": "MEDIO"},
            {"empresa_id": "E3", "predio_id": 7, "riesgo_total": "BAJO"},
            # E4 sin proveedores
        ])

    def test_classification(self):
        # Comentario: prueba que se clasifiquen correctamente según la proporción de proveedores con riesgo ALTO
        result = calculate_risk_enterprise(self.empresas_df, self.proveedores_df)
        e1 = result[result['empresa_id'] == "E1"].iloc[0]
        e2 = result[result['empresa_id'] == "E2"].iloc[0]
        e3 = result[result['empresa_id'] == "E3"].iloc[0]
        e4 = result[result['empresa_id'] == "E4"].iloc[0]

        self.assertEqual(e1['riesgo_total_empresa'], 'ALTO')
        self.assertEqual(e2['riesgo_total_empresa'], 'BAJO')
        self.assertEqual(e3['riesgo_total_empresa'], 'MEDIO')
        self.assertEqual(e4['riesgo_total_empresa'], 'SIN_DATOS')

if __name__ == "__main__":
    unittest.main()