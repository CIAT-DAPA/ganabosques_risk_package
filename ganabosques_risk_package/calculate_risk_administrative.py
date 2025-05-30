"""
Module: calculate_risk_administrative
Description: Aggregates risk information by administrative level (adm1, adm2, adm3)
using both farm and enterprise risk levels. Returns unified structure per record.
"""

import pandas as pd
from tqdm import tqdm
from ganabosques_risk_package.risk_level import RiskLevel, classify_risk

def summarize_admin(df, group_cols, score_col_prefix):
    """
    Groups and averages risk scores by the specified administrative columns.
    """
    grouped = df.groupby(group_cols)["risk_score"].mean().reset_index()
    grouped["risk_enum"] = grouped["risk_score"].apply(classify_risk)
    rename_map = {
        group_cols[0]: group_cols[0],
        group_cols[1]: group_cols[1],
        "risk_score": f"{score_col_prefix}_score",
        "risk_enum": f"{score_col_prefix}_risk"
    }
    return grouped.rename(columns=rename_map)

def merge_average(df1, df2, level):
    """
    Merges two risk dataframes (from plots and enterprises), averages scores and classifies final risk.
    """
    merged = pd.merge(
        df1,
        df2,
        on=[col for col in df1.columns if col.endswith("_id") or col.endswith("_name")],
        suffixes=('_plot', '_ent'),
        how='outer'
    ).fillna(0)

    merged[f"{level}_score"] = (merged[f"{level}_score_plot"] + merged[f"{level}_score_ent"]) / 2
    merged[f"{level}_risk"] = merged[f"{level}_score"].apply(classify_risk)

    return merged[[col for col in merged.columns if not col.endswith("_plot") and not col.endswith("_ent")]]

def calculate_risk_administrative(df_plots, df_enterprises):
    """
    Calculates the average risk per administrative level (adm1, adm2, adm3)
    based on farm and enterprise risks.

    Returns:
    - DataFrame with columns:
        adm1_id, adm1_name, adm1_score, adm1_risk,
        adm2_id, adm2_name, adm2_score, adm2_risk,
        adm3_id, adm3_name, adm3_score, adm3_risk
    """

    # Prepare working copies
    df_plots = df_plots.copy()
    df_enterprises = df_enterprises.copy()

    # Convert enum to score
    df_plots["risk_score"] = df_plots["risk_total"].apply(lambda x: x.value if isinstance(x, RiskLevel) else 0)
    df_enterprises["risk_score"] = df_enterprises["enterprise_risk_enum"].apply(lambda x: x.value if isinstance(x, RiskLevel) else 0)

    # Aggregate by level
    adm1_plots = summarize_admin(df_plots, ["adm1_id", "adm1_name"], "adm1")
    adm2_plots = summarize_admin(df_plots, ["adm1_id", "adm1_name", "adm2_id", "adm2_name"], "adm2")
    adm3_plots = summarize_admin(df_plots, ["adm1_id", "adm1_name", "adm2_id", "adm2_name", "adm3_id", "adm3_name"], "adm3")

    adm1_ent = summarize_admin(df_enterprises, ["adm1_id", "adm1_name"], "adm1")
    adm2_ent = summarize_admin(df_enterprises, ["adm1_id", "adm1_name", "adm2_id", "adm2_name"], "adm2")
    adm3_ent = summarize_admin(df_enterprises, ["adm1_id", "adm1_name", "adm2_id", "adm2_name", "adm3_id", "adm3_name"], "adm3")

    # Merge and average
    df_adm1 = merge_average(adm1_plots, adm1_ent, "adm1")
    df_adm2 = merge_average(adm2_plots, adm2_ent, "adm2")
    df_adm3 = merge_average(adm3_plots, adm3_ent, "adm3")

    # Final hierarchical join
    result = pd.merge(df_adm3, df_adm2, on=["adm1_id", "adm1_name", "adm2_id", "adm2_name"], how="left")
    result = pd.merge(result, df_adm1, on=["adm1_id", "adm1_name"], how="left")

    return result
