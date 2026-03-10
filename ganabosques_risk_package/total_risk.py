# -*- coding: utf-8 -*-
"""
Consolidación de riesgo total por predio.

Combina las alertas directas, indirectas y métricas espaciales en un solo
DataFrame por predio, generando la vista unificada de riesgo.

Columnas de salida:
  - id, direct_alert, deforested_ha, deforested_prop (de alertas directas)
  - n_in, n_out, n_indirect_in, n_indirect_out, n_total_mov,
    indirect_alert_in, indirect_alert_out (de alertas indirectas)
  - total_ha, farming_in_ha, farming_in_prop, farming_out_ha,
    farming_out_prop, protected_ha, protected_prop (de métricas espaciales)

Public API:
  - total_risk(direct_df, indirect_df=None, metrics_df=None) -> pd.DataFrame

"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DIRECT_COLS = ["id", "direct_alert", "deforested_ha", "deforested_prop"]

_INDIRECT_COLS = [
    "n_total_mov", "n_in", "n_out",
    "n_indirect_in", "n_indirect_out",
    "indirect_alert_in", "indirect_alert_out",
]

_METRICS_COLS = [
    "total_ha",
    "farming_in_ha", "farming_in_prop",
    "farming_out_ha", "farming_out_prop",
    "protected_ha", "protected_prop",
]


def _str_bool(x) -> bool:
    """Convierte valor a booleano."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes")
    try:
        return bool(x)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def total_risk(
    direct_df: pd.DataFrame,
    indirect_df: Optional[pd.DataFrame] = None,
    metrics_df: Optional[pd.DataFrame] = None,
    id_column: str = "id",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Consolida alertas directas, indirectas y métricas en un DataFrame unificado.

    Este es el resultado «total» que luego se pasa a ``alert_enterprise()``
    y ``supplier_risk()`` para determinar el riesgo de las empresas.

    Parameters
    ----------
    direct_df : pandas.DataFrame
        Resultado de ``alert_direct()`` con al menos: id, direct_alert,
        deforested_ha, deforested_prop. Puede incluir columnas de métricas
        si se pasó metrics_df a alert_direct(); en ese caso no es necesario
        pasar metrics_df aquí.
    indirect_df : pandas.DataFrame, optional
        Resultado de ``alert_indirect()`` con: id, n_in, n_out,
        n_indirect_in, n_indirect_out, n_total_mov, indirect_alert_in,
        indirect_alert_out.
        Si None, las columnas indirectas se llenan con "no_info".
    metrics_df : pandas.DataFrame, optional
        Resultado de ``spatial_metrics()`` con: id, total_ha,
        farming_in_ha, farming_in_prop, farming_out_ha, farming_out_prop,
        protected_ha, protected_prop.
        No es necesario si ya se incluyeron en direct_df.
    id_column : str, default "id"
        Nombre de la columna de ID.
    show_progress : bool, default True
        Mostrar resumen.

    Returns
    -------
    pandas.DataFrame
        Una fila por predio con todas las columnas combinadas:
        directas + indirectas + métricas.

    Examples
    --------
    >>> direct = alert_direct(plots, raster, metrics_df=metrics)
    >>> indirect = alert_indirect(direct, movements)
    >>> risk = total_risk(direct, indirect)
    """
    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Validar direct_df
    # -------------------------------------------------------------------------
    if id_column not in direct_df.columns:
        raise ValueError(f"direct_df debe contener columna '{id_column}'.")

    if "direct_alert" not in direct_df.columns:
        raise ValueError("direct_df debe contener columna 'direct_alert'.")

    # Asegurar id como string
    df = direct_df.copy()
    df[id_column] = df[id_column].astype(str)

    # Asegurar direct_alert como booleano
    df["direct_alert"] = df["direct_alert"].map(_str_bool)

    # Asegurar columnas directas
    for c in ["deforested_ha", "deforested_prop"]:
        if c not in df.columns:
            df[c] = 0.0

    if show_progress:
        n = len(df)
        n_alerted = df["direct_alert"].sum()
        print(f"📊 Consolidando riesgo total para {n:,} predios "
              f"({n_alerted:,} con alerta directa)")

    # -------------------------------------------------------------------------
    # Merge con indirectas
    # -------------------------------------------------------------------------
    if indirect_df is not None and not indirect_df.empty:
        ind = indirect_df.copy()
        ind[id_column] = ind[id_column].astype(str)

        # Asegurar columnas indirectas existen
        for c in _INDIRECT_COLS:
            if c not in ind.columns:
                ind[c] = "no_info"

        df = pd.merge(
            df,
            ind[[id_column] + _INDIRECT_COLS],
            on=id_column, how="outer", copy=False,
        )
    else:
        # Sin datos indirectos: llenar con "no_info"
        for c in _INDIRECT_COLS:
            if c not in df.columns:
                df[c] = "no_info"

    # -------------------------------------------------------------------------
    # Merge con métricas espaciales (si no están ya en direct_df)
    # -------------------------------------------------------------------------
    has_metrics_in_df = all(c in df.columns for c in _METRICS_COLS)

    if metrics_df is not None and not metrics_df.empty and not has_metrics_in_df:
        met = metrics_df.copy()
        met[id_column] = met[id_column].astype(str)
        df = pd.merge(df, met, on=id_column, how="left", copy=False)

    # Asegurar columnas de métricas existen (por si no se pasó nada)
    for c in _METRICS_COLS:
        if c not in df.columns:
            df[c] = 0.0

    # -------------------------------------------------------------------------
    # Resultado final
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - t0

    if show_progress:
        print(f"✅ Riesgo total consolidado: {len(df):,} predios en {elapsed:.2f}s")

        # Resumen rápido
        try:
            n_direct = df["direct_alert"].sum()
            n_in = (df["indirect_alert_in"] == True).sum() if "indirect_alert_in" in df.columns else 0  # noqa
            n_out = (df["indirect_alert_out"] == True).sum() if "indirect_alert_out" in df.columns else 0  # noqa
            print(f"   📊 Alerta directa: {n_direct:,}")
            print(f"   📊 Alerta indirecta IN: {n_in:,}")
            print(f"   📊 Alerta indirecta OUT: {n_out:,}")
        except Exception:
            pass

    return df
