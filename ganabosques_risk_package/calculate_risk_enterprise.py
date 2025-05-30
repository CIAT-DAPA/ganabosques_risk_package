"""
Module: calculate_risk_enterprise
Description: Evaluates total risk for enterprises based on supplier risk levels using MRV criteria.
"""

import pandas as pd

def calculate_risk_enterprise(empresas_df, proveedores_df):
    """
    Evaluate risk level of each enterprise based on its suppliers' risk data.

    Parameters:
    - empresas_df: DataFrame with column 'empresa_id'
    - proveedores_df: DataFrame with columns:
        ['empresa_id', 'plot_id', 'risk_direct_level', 'riesgo_total']

    Returns:
    - DataFrame with one row per empresa and risk classification + indicators
    """

    def clasificar_empresa(subdf):
        total = len(subdf)
        if total == 0:
            return 'SIN_DATOS', 0, 0, 0

        count_alto = sum(subdf['riesgo_total'] == 'ALTO')
        count_medio = sum(subdf['riesgo_total'] == 'MEDIO')
        count_bajo = sum(subdf['riesgo_total'] == 'BAJO')

        perc_alto = count_alto / total * 100

        if perc_alto >= 5:
            nivel = 'ALTO'
        elif perc_alto > 0:
            nivel = 'MEDIO'
        else:
            nivel = 'BAJO'

        return nivel, count_alto, count_medio, count_bajo

    resultados = []
    for eid in empresas_df['empresa_id'].unique():
        subdf = proveedores_df[proveedores_df['empresa_id'] == eid]
        nivel, alto, medio, bajo = clasificar_empresa(subdf)
        resultados.append({
            'empresa_id': eid,
            'riesgo_total_empresa': nivel,
            'n_proveedores_alto': alto,
            'n_proveedores_medio': medio,
            'n_proveedores_bajo': bajo
        })

    return pd.DataFrame(resultados)