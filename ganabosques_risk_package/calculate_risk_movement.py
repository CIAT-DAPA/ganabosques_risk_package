"""
Module: calculate_risk_movement
Description: Calculates entry and exit movement risks for farms based on the MRV protocol.
"""

import pandas as pd
from tqdm import tqdm
from ganabosques_risk_package.risk_level import RiskLevel

def compute_average(values):
    """
    Calculate the average of a list of numeric values.
    """
    return sum(values) / len(values) if values else 0

def classify_risk(score):
    """
    Classify the average score into a RiskLevel category.
    """
    if score >= 2.5:
        return RiskLevel.HIGH
    elif score >= 1.5:
        return RiskLevel.MEDIUM
    elif score > 0:
        return RiskLevel.LOW
    else:
        return RiskLevel.NO_RISK

def calculate_risk_movement(df_plots_risk, df_movement):
    """
    Calculate average entry and exit risks for multiple plots based on movement data.

    Parameters:
    - df_plots_risk: DataFrame from calculate_risk_direct, must include 'id' and 'risk_direct_level'
    - df_movement: DataFrame with columns ['origen_id', 'destination_id']

    Returns:
    - DataFrame with columns:
        - plot_id
        - riesgo_entrada_enum
        - riesgo_entrada_label
        - riesgo_salida_enum
        - riesgo_salida_label
    """
    if not {'origen_id', 'destination_id'}.issubset(df_movement.columns):
        raise ValueError("df_movement must contain columns: origen_id, destination_id")

    if not {'id', 'risk_direct_level'}.issubset(df_plots_risk.columns):
        raise ValueError("df_plots_risk must contain columns: id, risk_direct_level")

    plot_risk_dict = dict(zip(df_plots_risk['id'], df_plots_risk['risk_direct_level']))

    results = []

    for plot_id in tqdm(df_plots_risk['id'], desc="Calculating movement risk"):
        in_raw = df_movement[df_movement['destination_id'] == plot_id]['origen_id']
        out_raw = df_movement[df_movement['origen_id'] == plot_id]['destination_id']

        in_scores = [
            plot_risk_dict.get(ent).value if ent in plot_risk_dict and isinstance(plot_risk_dict.get(ent), RiskLevel) else 0 for ent in in_raw
        ]
        out_scores = [
            plot_risk_dict.get(sal).value if sal in plot_risk_dict and isinstance(plot_risk_dict.get(sal), RiskLevel) else 0 for sal in out_raw
        ]

        avg_in = compute_average(in_scores)
        avg_out = compute_average(out_scores)

        riesgo_entrada_enum = classify_risk(avg_in)
        riesgo_salida_enum = classify_risk(avg_out)

        results.append({
            "plot_id": plot_id,
            "riesgo_entrada_enum": riesgo_entrada_enum,
            "riesgo_entrada_label": riesgo_entrada_enum.value,
            "riesgo_salida_enum": riesgo_salida_enum,
            "riesgo_salida_label": riesgo_salida_enum.value
        })

    return pd.DataFrame(results)
