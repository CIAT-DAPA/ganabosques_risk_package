"""
Module: calculate_risk_movement
Description: Calculates entry and exit movement risks for farms based on the MRV protocol.
"""

import pandas as pd
from tqdm import tqdm
from ganabosques_risk_package.risk_level import RiskLevel, classify_risk

def compute_average(values):
    """
    Calculate the average of a list of numeric values.
    """
    return sum(values) / len(values) if values else 0

def calculate_risk_movement(df_plots_risk, df_movement):
    """
    Calculate average entry and exit risks for multiple plots based on movement data.

    Parameters:
    - df_plots_risk: DataFrame with columns:
        - 'id': unique identifier of the plot
        - 'risk_direct_level': RiskLevel enum instance from direct risk calculation
    - df_movement: DataFrame with movement records. Must include:
        - 'origen_id': origin plot ID of the movement
        - 'destination_id': destination plot ID of the movement

    Returns:
    - DataFrame with columns:
        - plot_id: unique ID of the evaluated plot
        - risk_in_score: numeric average of entry risk scores
        - risk_in_enum: RiskLevel enum for entry
        - risk_in_enum_value: numeric value of the enum (e.g., 3 for HIGH)
        - risk_in_enum_label: string name of the enum (e.g., "HIGH")
        - risk_out_score: numeric average of exit risk scores
        - risk_out_enum: RiskLevel enum for exit
        - risk_out_enum_value: numeric value of the enum
        - risk_out_enum_label: string name of the enum
    """

    # Validate input columns
    if not {'origen_id', 'destination_id'}.issubset(df_movement.columns):
        raise ValueError("df_movement must contain columns: origen_id, destination_id")

    if not {'id', 'risk_direct_level'}.issubset(df_plots_risk.columns):
        raise ValueError("df_plots_risk must contain columns: id, risk_direct_level")

    # Create lookup dictionary: plot_id -> RiskLevel
    plot_risk_dict = dict(zip(df_plots_risk['id'], df_plots_risk['risk_direct_level']))

    results = []

    # Iterate over each plot and calculate entry/exit risk scores
    for plot_id in tqdm(df_plots_risk['id'], desc="Calculating movement risk " + str(len(df_plots_risk)) + " plots"):

        # Find all incoming and outgoing movements for the plot
        in_raw = df_movement[df_movement['destination_id'] == plot_id]['origen_id']
        out_raw = df_movement[df_movement['origen_id'] == plot_id]['destination_id']

        # Convert origin/destination plot risks into numeric scores
        in_scores = [
            plot_risk_dict.get(ent).value if ent in plot_risk_dict and isinstance(plot_risk_dict.get(ent), RiskLevel) else 0
            for ent in in_raw
        ]
        out_scores = [
            plot_risk_dict.get(sal).value if sal in plot_risk_dict and isinstance(plot_risk_dict.get(sal), RiskLevel) else 0
            for sal in out_raw
        ]

        # Compute average scores
        avg_in = compute_average(in_scores)
        avg_out = compute_average(out_scores)

        # Classify average scores into RiskLevel enums
        risk_in_enum = classify_risk(avg_in)
        risk_out_enum = classify_risk(avg_out)

        # Store results for this plot
        results.append({
            "plot_id": plot_id,
            "risk_in_score": avg_in,
            "risk_in_enum": risk_in_enum,
            "risk_in_enum_value": risk_in_enum.value,
            "risk_in_enum_label": risk_in_enum.name,
            "risk_out_score": avg_out,
            "risk_out_enum": risk_out_enum,
            "risk_out_enum_value": risk_out_enum.value,
            "risk_out_enum_label": risk_out_enum.name,
        })

    return pd.DataFrame(results)

