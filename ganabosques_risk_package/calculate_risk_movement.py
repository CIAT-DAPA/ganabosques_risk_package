"""
Module: calculate_risk_movement
Description: Calculates entry and exit movement risks for a given farm based on the MRV protocol.
"""

import pandas as pd

def calculate_risk_movement(predio_id, movilizaciones_df):
    """
    Calculate movement risks (entrada/salida) for a specific farm.

    Parameters:
    - predio_id: identifier of the farm to analyze
    - movilizaciones_df: DataFrame with columns:
        ['id_origen', 'id_destino', 'tipo', 'nivel_riesgo']
        where tipo is either 'entrada' or 'salida'

    Returns:
    - Dictionary with keys:
        - 'predio_id'
        - 'riesgo_entrada_promedio'
        - 'riesgo_salida_promedio'
    """

    if not {'id_origen', 'id_destino', 'tipo', 'nivel_riesgo'}.issubset(movilizaciones_df.columns):
        raise ValueError("movilizaciones_df must contain columns: id_origen, id_destino, tipo, nivel_riesgo")

    niveles = {'BAJO': 1, 'MEDIO': 2, 'ALTO': 3}
    riesgos = {'entrada': [], 'salida': []}

    for _, row in movilizaciones_df.iterrows():
        if row['tipo'] == 'entrada' and row['id_destino'] == predio_id:
            riesgos['entrada'].append(niveles.get(row['nivel_riesgo'].upper(), 0))
        elif row['tipo'] == 'salida' and row['id_origen'] == predio_id:
            riesgos['salida'].append(niveles.get(row['nivel_riesgo'].upper(), 0))

    def promedio(valores):
        return sum(valores) / len(valores) if valores else 0

    def clasificar(n):
        if n >= 2.5:
            return 'ALTO'
        elif n >= 1.5:
            return 'MEDIO'
        else:
            return 'BAJO'

    entrada_valor = promedio(riesgos['entrada'])
    salida_valor = promedio(riesgos['salida'])

    return {
        'predio_id': predio_id,
        'riesgo_entrada_promedio': clasificar(entrada_valor),
        'riesgo_salida_promedio': clasificar(salida_valor)
    }