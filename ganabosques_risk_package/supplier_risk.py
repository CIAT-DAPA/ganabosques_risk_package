# -*- coding: utf-8 -*-
"""
Riesgo de empresa basado en relaciones Supplier (empresa↔finca).

A diferencia de ``alert_enterprise`` (que usa movimientos de ganado), este
módulo calcula el riesgo de empresas que **no** tienen movimientos pero **sí**
tienen relación formal (supplier) con fincas.

Recibe DataFrames puros.

Public API:
  - supplier_risk(total_risk_df, suppliers_df, ...) -> pd.DataFrame

Helpers:
  - get_years_for_period(period, period_type) -> set[int]
  - filter_suppliers_by_period(suppliers_df, period, period_type) -> pd.DataFrame

"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional, Set

import pandas as pd


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _normalize_id(value) -> str:
    """Normaliza un ID a string limpio."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    if re.match(r"^\d+\.0$", s):
        s = s[:-2]
    if re.match(r"^0*\d+$", s):
        s = str(int(s))
    return s.upper()


def _str_bool(x) -> bool:
    """Convierte valor a booleano."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "y", "si", "sí", "t")
    try:
        return bool(x)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Helpers de períodos (portados del calculador original)
# ---------------------------------------------------------------------------

def get_years_for_period(period: str, period_type: str) -> Set[int]:
    """Extrae el conjunto de años que aplican para un período.

    Parameters
    ----------
    period : str
        Código del período (``"2017"``, ``"2010-2024"``, ``"201701"``).
    period_type : str
        Tipo: ``"annual"``, ``"cumulative"``, ``"nad"``, ``"atd"``.

    Returns
    -------
    set of int

    Examples
    --------
    >>> get_years_for_period("2010-2024", "cumulative")
    {2010, 2011, ..., 2024}
    >>> get_years_for_period("201701", "nad")
    {2017}
    """
    if period_type == "cumulative" and "-" in period:
        parts = period.split("-")
        return set(range(int(parts[0]), int(parts[1]) + 1))
    elif period_type in ("nad", "atd") and len(period) == 6:
        return {int(period[:4])}
    elif period_type == "annual" and "-" in period:
        parts = period.split("-")
        return set(range(int(parts[0]), int(parts[1]) + 1))
    else:
        try:
            return {int(period[:4])}
        except ValueError:
            return set()


def filter_suppliers_by_period(
    suppliers_df: pd.DataFrame,
    period: str,
    period_type: str,
    years_column: str = "years",
    filter_by_year: bool = False,
) -> pd.DataFrame:
    """Filtra suppliers cuyo rango de años intersecta con el período.

    Parameters
    ----------
    suppliers_df : pandas.DataFrame
        DataFrame con al menos ``enterprise_id``, ``farm_id`` y
        la columna de años.
    period : str
        Código del período.
    period_type : str
        Tipo de período.
    years_column : str, default "years"
        Columna que contiene los años de relación (lista de int o string
        separado por comas).
    filter_by_year : bool, default False
        Si True, solo incluye suppliers cuyo rango de años intersecta
        con los años del período. Si False, incluye todos (comportamiento
        actual del calculador original para corrida inicial).

    Returns
    -------
    pandas.DataFrame
        Subconjunto de suppliers_df que aplican.
    """
    if not filter_by_year:
        return suppliers_df.copy()

    target_years = get_years_for_period(period, period_type)
    if not target_years:
        return suppliers_df.iloc[0:0].copy()

    def _matches(years_val) -> bool:
        if isinstance(years_val, (list, set, tuple)):
            return bool(set(int(y) for y in years_val) & target_years)
        if isinstance(years_val, str):
            try:
                parsed = {int(y.strip()) for y in years_val.split(",") if y.strip()}
                return bool(parsed & target_years)
            except ValueError:
                return False
        if isinstance(years_val, (int, float)) and not pd.isna(years_val):
            return int(years_val) in target_years
        return False

    mask = suppliers_df[years_column].apply(_matches)
    return suppliers_df[mask].copy()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def supplier_risk(
    total_risk_df: pd.DataFrame,
    suppliers_df: pd.DataFrame,
    period: Optional[str] = None,
    period_type: Optional[str] = None,
    id_column: str = "id",
    farm_id_column: str = "farm_id",
    enterprise_id_column: str = "enterprise_id",
    years_column: str = "years",
    filter_by_year: bool = False,
    alert_columns: Optional[List[str]] = None,
    normalize_ids: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Calcula riesgo de empresa basado en relaciones supplier.

    Cruza la tabla de suppliers (empresa↔finca) con el riesgo total de las
    fincas para identificar qué empresas tienen fincas con alerta.

    Parameters
    ----------
    total_risk_df : pandas.DataFrame
        Resultado de ``total_risk()`` (o ``alert_direct()``). Debe contener
        ``id_column`` y al menos una columna de alerta.
    suppliers_df : pandas.DataFrame
        Relaciones empresa-finca. Columnas mínimas:

        - ``enterprise_id`` : ID de la empresa.
        - ``farm_id`` : ID de la finca.
        - ``years`` (opcional) : Años de relación (lista, CSV, o int).

        Puede incluir metadatos extra (``enterprise_name``,
        ``enterprise_type``, etc.) que se preservarán en la salida.
    period : str, optional
        Código del período (solo necesario si ``filter_by_year=True``).
    period_type : str, optional
        Tipo de período (solo necesario si ``filter_by_year=True``).
    id_column : str, default "id"
        Columna de ID de finca en ``total_risk_df``.
    farm_id_column : str, default "farm_id"
        Columna de ID de finca en ``suppliers_df``.
    enterprise_id_column : str, default "enterprise_id"
        Columna de ID de empresa en ``suppliers_df``.
    years_column : str, default "years"
        Columna de años en suppliers_df.
    filter_by_year : bool, default False
        Si True, filtra suppliers por intersección de años con el período.
    alert_columns : list of str, optional
        Columnas booleanas que definen alerta.
        Default: ``["direct_alert", "indirect_alert_in", "indirect_alert_out"]``.
    normalize_ids : bool, default True
        Normalizar IDs.
    show_progress : bool, default True
        Mostrar resumen.

    Returns
    -------
    pandas.DataFrame
        Una fila por relación enterprise-farm que tiene riesgo activo.
        Columnas: ``enterprise_id, farm_id, has_risk, risk_direct,
        risk_indirect_in, risk_indirect_out, deforested_ha,
        deforested_prop`` + cualquier columna extra de suppliers_df.

    Examples
    --------
    >>> risk = total_risk(direct_df, indirect_df, metrics_df)
    >>> suppliers = pd.read_csv("suppliers.csv")
    >>> ent_risk = supplier_risk(risk, suppliers)
    >>> ent_risk.groupby("enterprise_id")["has_risk"].sum()
    """
    t0 = time.perf_counter()

    if alert_columns is None:
        alert_columns = ["direct_alert", "indirect_alert_in", "indirect_alert_out"]

    # ------------------------------------------------------------------
    # 1. Validar inputs
    # ------------------------------------------------------------------
    if id_column not in total_risk_df.columns:
        raise ValueError(f"total_risk_df debe contener columna '{id_column}'.")
    if farm_id_column not in suppliers_df.columns:
        raise ValueError(f"suppliers_df debe contener columna '{farm_id_column}'.")
    if enterprise_id_column not in suppliers_df.columns:
        raise ValueError(f"suppliers_df debe contener columna '{enterprise_id_column}'.")

    # ------------------------------------------------------------------
    # 2. Filtrar suppliers por período (si aplica)
    # ------------------------------------------------------------------
    if filter_by_year and period and period_type:
        if years_column not in suppliers_df.columns:
            logging.warning(
                f"filter_by_year=True pero columna '{years_column}' no existe. "
                f"Se usan todos los suppliers."
            )
            sups = suppliers_df.copy()
        else:
            sups = filter_suppliers_by_period(
                suppliers_df, period, period_type, years_column, filter_by_year=True
            )
    else:
        sups = suppliers_df.copy()

    if sups.empty:
        if show_progress:
            print("⚠️  No hay suppliers que apliquen para este período.")
        return pd.DataFrame(columns=[
            enterprise_id_column, farm_id_column,
            "has_risk", "risk_direct", "risk_indirect_in", "risk_indirect_out",
            "deforested_ha", "deforested_prop",
        ])

    # ------------------------------------------------------------------
    # 3. Preparar IDs
    # ------------------------------------------------------------------
    risk = total_risk_df.copy()
    risk[id_column] = risk[id_column].astype(str)
    sups[farm_id_column] = sups[farm_id_column].astype(str)
    sups[enterprise_id_column] = sups[enterprise_id_column].astype(str)

    if normalize_ids:
        risk[id_column] = risk[id_column].apply(_normalize_id)
        sups[farm_id_column] = sups[farm_id_column].apply(_normalize_id)

    # ------------------------------------------------------------------
    # 4. Determinar alertas por finca
    # ------------------------------------------------------------------
    existing_alert_cols = [c for c in alert_columns if c in risk.columns]

    if not existing_alert_cols:
        logging.warning(
            f"No se encontraron columnas de alerta en total_risk_df. "
            f"Buscadas: {alert_columns}"
        )

    # Crear columnas de riesgo individuales
    risk["_risk_direct"] = (
        risk["direct_alert"].map(_str_bool) if "direct_alert" in risk.columns
        else False
    )
    risk["_risk_indirect_in"] = (
        risk["indirect_alert_in"].map(_str_bool) if "indirect_alert_in" in risk.columns
        else False
    )
    risk["_risk_indirect_out"] = (
        risk["indirect_alert_out"].map(_str_bool) if "indirect_alert_out" in risk.columns
        else False
    )
    risk["_has_risk"] = (
        risk["_risk_direct"] | risk["_risk_indirect_in"] | risk["_risk_indirect_out"]
    )

    # Columnas relevantes para el merge
    risk_cols = [id_column, "_has_risk", "_risk_direct",
                 "_risk_indirect_in", "_risk_indirect_out"]
    for extra in ["deforested_ha", "deforested_prop"]:
        if extra in risk.columns:
            risk_cols.append(extra)
    risk_slim = risk[risk_cols].copy()

    if show_progress:
        n_sup = len(sups)
        n_ent = sups[enterprise_id_column].nunique()
        n_farms = sups[farm_id_column].nunique()
        print(f"🔗 Evaluando riesgo de {n_ent:,} empresas con "
              f"{n_farms:,} fincas ({n_sup:,} relaciones)")

    # ------------------------------------------------------------------
    # 5. Merge suppliers ↔ riesgo de fincas
    # ------------------------------------------------------------------
    merged = pd.merge(
        sups,
        risk_slim,
        left_on=farm_id_column,
        right_on=id_column,
        how="left",
    )

    # Si id_column != farm_id_column, drop la columna duplicada
    if id_column != farm_id_column and id_column in merged.columns:
        merged = merged.drop(columns=[id_column])

    # Rellenar fincas que no estaban en total_risk_df (sin info de riesgo)
    merged["_has_risk"] = merged["_has_risk"].fillna(False)
    merged["_risk_direct"] = merged["_risk_direct"].fillna(False)
    merged["_risk_indirect_in"] = merged["_risk_indirect_in"].fillna(False)
    merged["_risk_indirect_out"] = merged["_risk_indirect_out"].fillna(False)

    # Renombrar columnas internas
    merged = merged.rename(columns={
        "_has_risk": "has_risk",
        "_risk_direct": "risk_direct",
        "_risk_indirect_in": "risk_indirect_in",
        "_risk_indirect_out": "risk_indirect_out",
    })

    # ------------------------------------------------------------------
    # 6. Filtrar solo relaciones con riesgo activo
    # ------------------------------------------------------------------
    result = merged[merged["has_risk"]].copy()

    elapsed = time.perf_counter() - t0

    if show_progress:
        n_risky = len(result)
        n_ent_risky = result[enterprise_id_column].nunique() if not result.empty else 0
        n_farms_risky = result[farm_id_column].nunique() if not result.empty else 0
        print(f"✅ Riesgo por suppliers: {n_risky:,} relaciones con riesgo "
              f"en {elapsed:.2f}s")
        print(f"   📊 Empresas con riesgo: {n_ent_risky:,}")
        print(f"   📊 Fincas con riesgo: {n_farms_risky:,}")
        if n_risky > 0:
            n_direct = result["risk_direct"].sum()
            n_ind_in = result["risk_indirect_in"].sum()
            n_ind_out = result["risk_indirect_out"].sum()
            print(f"   📊 Por tipo — directa: {n_direct:,}, "
                  f"indirecta IN: {n_ind_in:,}, indirecta OUT: {n_ind_out:,}")

    return result.reset_index(drop=True)


def supplier_risk_summary(
    supplier_risk_df: pd.DataFrame,
    enterprise_id_column: str = "enterprise_id",
) -> pd.DataFrame:
    """Genera resumen agregado por empresa del resultado de ``supplier_risk()``.

    Parameters
    ----------
    supplier_risk_df : pandas.DataFrame
        Resultado de ``supplier_risk()``.
    enterprise_id_column : str, default "enterprise_id"
        Columna de ID de empresa.

    Returns
    -------
    pandas.DataFrame
        Una fila por empresa. Columnas: ``enterprise_id, n_farms_at_risk,
        n_risk_direct, n_risk_indirect_in, n_risk_indirect_out,
        total_deforested_ha``.
    """
    if supplier_risk_df.empty:
        return pd.DataFrame(columns=[
            enterprise_id_column, "n_farms_at_risk",
            "n_risk_direct", "n_risk_indirect_in", "n_risk_indirect_out",
            "total_deforested_ha",
        ])

    agg = supplier_risk_df.groupby(enterprise_id_column).agg(
        n_farms_at_risk=("has_risk", "sum"),
        n_risk_direct=("risk_direct", "sum"),
        n_risk_indirect_in=("risk_indirect_in", "sum"),
        n_risk_indirect_out=("risk_indirect_out", "sum"),
        total_deforested_ha=("deforested_ha", lambda x: x.sum() if "deforested_ha" in supplier_risk_df.columns else 0),
    ).reset_index()

    return agg.sort_values("n_farms_at_risk", ascending=False).reset_index(drop=True)
