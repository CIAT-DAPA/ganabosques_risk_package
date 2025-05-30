"""
Module: calculate_risk_enterprise
Description: Calculates total enterprise risk based on supplier plot risks and compliance with MRV criteria.
"""

import pandas as pd
from tqdm import tqdm
from ganabosques_risk_package.risk_level import RiskLevel, classify_risk

def calculate_risk_enterprise(df_enterprises, df_suppliers, df_plots_risk):
    """
    Evaluate the risk level for each enterprise based on the MRV protocol criteria.

    Parameters:
    - df_enterprises: DataFrame with at least column ['id'] for enterprise ID
    - df_suppliers: DataFrame with columns ['enterprise_id', 'plot_id']
    - df_plots_risk: DataFrame with at least columns ['plot_id', 'risk_total'] from calculate_risk_total

    Returns:
    - DataFrame with one row per enterprise, including:
        - enterprise_id
        - enterprise_risk_enum
        - enterprise_risk_value
        - enterprise_risk_label
        - criteria_deforestation_high (bool)
        - criteria_location_restricted (bool)
        - criteria_municipality_critical (bool)
        - criteria_geolocation_coverage (bool)
    """

    if not {'id'}.issubset(df_enterprises.columns):
        raise ValueError("df_enterprises must have column 'id'")
    if not {'enterprise_id', 'plot_id'}.issubset(df_suppliers.columns):
        raise ValueError("df_suppliers must have columns 'enterprise_id' and 'plot_id'")
    if not {'plot_id', 'risk_total'}.issubset(df_plots_risk.columns):
        raise ValueError("df_plots_risk must have columns 'plot_id' and 'risk_total'")

    # Prepare dict with the plots risk
    risk_dict = dict(zip(df_plots_risk['plot_id'], df_plots_risk['risk_total']))

    results = []

    for _, row in tqdm(df_enterprises.iterrows(), total=len(df_enterprises), desc="Calculating enterprise risk " + str(len(df_enterprises)) + " enterprises"):
        enterprise_id = row['id']

        # Extract plots from this enterprise's suppliers
        supplier_plots = df_suppliers[df_suppliers['enterprise_id'] == enterprise_id]['plot_id'].tolist()
        num_total = len(supplier_plots)
        risk_values = [risk_dict.get(pid, RiskLevel.NO_RISK) for pid in supplier_plots if isinstance(risk_dict.get(pid), RiskLevel)]

        # Amount plots by each type of risk
        num_high = sum(1 for r in risk_values if r == RiskLevel.HIGH)
        num_medium = sum(1 for r in risk_values if r == RiskLevel.MEDIUM)
        num_low = sum(1 for r in risk_values if r == RiskLevel.LOW)
        num_with_location = len(risk_values)

        # Evaluar criterios
        criteria_deforestation_high = (num_total > 0 and (num_high / num_total) >= 0.05)
        criteria_location_restricted = any(r in (RiskLevel.HIGH, RiskLevel.MEDIUM) for r in risk_values)  # simplificación proxy
        criteria_municipality_critical = criteria_location_restricted  # usar como equivalente si no hay municipios
        coverage = (num_with_location / num_total) if num_total > 0 else 0
        criteria_geolocation_coverage = coverage >= 0.61

        # Asignar riesgo por reglas
        if criteria_deforestation_high or not criteria_geolocation_coverage or criteria_location_restricted or criteria_municipality_critical:
            if not criteria_geolocation_coverage or criteria_deforestation_high:
                risk = RiskLevel.HIGH
            elif coverage >= 0.41:
                risk = RiskLevel.MEDIUM
            else:
                risk = RiskLevel.HIGH
        else:
            risk = RiskLevel.LOW

        results.append({
            "enterprise_id": enterprise_id,
            "enterprise_risk": risk,
            "enterprise_risk_value": risk.value,
            "enterprise_risk_label": risk.name,
            "criteria_deforestation_high": criteria_deforestation_high,
            "criteria_location_restricted": criteria_location_restricted,
            "criteria_municipality_critical": criteria_municipality_critical,
            "criteria_geolocation_coverage": criteria_geolocation_coverage
        })

    return pd.DataFrame(results)
