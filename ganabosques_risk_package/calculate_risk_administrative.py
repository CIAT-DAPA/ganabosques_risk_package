"""
Module: calculate_risk_administrative
Description: Aggregates total risk per administrative unit (e.g., vereda).
"""

import pandas as pd

def calculate_risk_administrative(df_predios):
    """
    Aggregates total risk by vereda from predio-level risk data.

    Parameters:
    - df_predios: DataFrame with columns ['vereda', 'riesgo_total']

    Returns:
    - DataFrame with columns ['vereda', 'riesgo_vereda', 'n_predios', 'n_alto', 'n_medio', 'n_bajo']
    """

    if not {'vereda', 'riesgo_total'}.issubset(df_predios.columns):
        raise ValueError("df_predios must contain 'vereda' and 'riesgo_total' columns")

    resultados = []

    for vereda, group in df_predios.groupby("vereda"):
        n = len(group)
        alto = (group['riesgo_total'] == 'ALTO').sum()
        medio = (group['riesgo_total'] == 'MEDIO').sum()
        bajo = (group['riesgo_total'] == 'BAJO').sum()

        nivel = 'ALTO' if alto / n >= 0.5 else 'MEDIO' if medio / n >= 0.5 else 'BAJO'

        resultados.append({
            "vereda": vereda,
            "riesgo_vereda": nivel,
            "n_predios": n,
            "n_alto": alto,
            "n_medio": medio,
            "n_bajo": bajo
        })

    return pd.DataFrame(resultados)