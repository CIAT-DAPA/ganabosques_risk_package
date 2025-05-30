"""
Module: calculate_risk_total
Description: Computes the total risk level for each plot based on weighted components:
direct risk, entry risk, and exit risk, following the MRV protocol.
"""

import pandas as pd
from tqdm import tqdm
from ganabosques_risk_package.risk_level import RiskLevel, classify_risk

def calculate_risk_total(df_direct, df_movement):
    """
    Calculate the total risk level per plot, using 50% direct risk,
    40% movement-in risk, and 10% movement-out risk.

    Parameters:
    - df_direct: DataFrame from calculate_risk_direct, includes 'id' and 'risk_direct_level'
    - df_movement: DataFrame from calculate_risk_movement, includes:
        'plot_id', 'risk_in_score', 'risk_out_score'

    Returns:
    - DataFrame with:
        - plot_id
        - risk_direct_value
        - risk_in_score
        - risk_out_score
        - risk_total
        - risk_total_value
        - risk_total_label
        - risk_total_score
    """
    if not {'id', 'risk_direct_level'}.issubset(df_direct.columns):
        raise ValueError("df_direct must include 'id' and 'risk_direct_level' columns")
    if not {'plot_id', 'risk_in_score', 'risk_out_score'}.issubset(df_movement.columns):
        raise ValueError("df_movement must include 'plot_id', 'risk_in_score', 'risk_out_score'")

    df_movement_indexed = df_movement.set_index('plot_id')

    results = []

    for _, row in tqdm(df_direct.iterrows(), total=len(df_direct), desc="Calculating total risk " + str(len(df_direct)) + " plots"):
        plot_id = row['id']
        direct_enum = row['risk_direct_level']
        risk_direct_value = direct_enum.value

        # Get corresponding movement scores, default to 0 if not found
        risk_in_score = 0
        risk_out_score = 0

        if plot_id in df_movement_indexed.index:
            mov_row = df_movement_indexed.loc[plot_id]
            risk_in_score = mov_row['risk_in_score']
            risk_out_score = mov_row['risk_out_score']

        # Compute weighted total score
        total_score = (0.5 * risk_direct_value) + (0.4 * risk_in_score) + (0.1 * risk_out_score)
        total_enum = classify_risk(total_score)

        results.append({
            "plot_id": plot_id,
            "risk_direct_value": risk_direct_value,
            "risk_in_score": risk_in_score,
            "risk_out_score": risk_out_score,
            "risk_total": total_enum,
            "risk_total_value": total_enum.value,
            "risk_total_label": total_enum.name,
            "risk_total_score": total_score
        })

    return pd.DataFrame(results)
