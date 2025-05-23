"""
Module: calculate_risk_total
Description: Aggregates direct, entry, and exit risks into a total risk classification.
"""

import pandas as pd

def calculate_risk_total(df_predios):
    """
    Combines risk_direct_level, riesgo_entrada_promedio and riesgo_salida_promedio into one total risk.

    Parameters:
    - df_predios: DataFrame with columns:
        ['id', 'risk_direct_level', 'riesgo_entrada_promedio', 'riesgo_salida_promedio']

    Returns:
    - DataFrame with column 'riesgo_total' added
    """
    if not {'id', 'risk_direct_level', 'riesgo_entrada_promedio', 'riesgo_salida_promedio'}.issubset(df_predios.columns):
        raise ValueError("df_predios must contain 'id', 'risk_direct_level', 'riesgo_entrada_promedio', 'riesgo_salida_promedio' columns")

    niveles = {'BAJO': 1, 'MEDIO': 2, 'ALTO': 3}
    clasificar = lambda n: 'ALTO' if n >= 2.5 else 'MEDIO' if n >= 1.5 else 'BAJO'

    riesgos = []
    for _, row in df_predios.iterrows():
        valores = [
            niveles.get(row['risk_direct_level'].upper(), 1),
            niveles.get(row['riesgo_entrada_promedio'].upper(), 1),
            niveles.get(row['riesgo_salida_promedio'].upper(), 1)
        ]
        riesgo_total = clasificar(sum(valores) / len(valores))
        riesgos.append(riesgo_total)

    df_predios = df_predios.copy()
    df_predios['riesgo_total'] = riesgos
    return df_predios