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

from ._utils import echo, str_bool as _str_bool


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


_INDIRECT_BOOL_COLS = ["indirect_alert_in", "indirect_alert_out"]


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
        echo(f"📊 Consolidando riesgo total para {n:,} predios "
              f"({n_alerted:,} con alerta directa)")

    # -------------------------------------------------------------------------
    # Merge con indirectas
    # -------------------------------------------------------------------------
    if indirect_df is not None and not indirect_df.empty:
        ind = indirect_df.copy()
        ind[id_column] = ind[id_column].astype(str)
        provided_cols = [c for c in _INDIRECT_COLS if c in ind.columns]

        # Asegurar columnas indirectas existen
        for c in _INDIRECT_COLS:
            if c not in ind.columns:
                ind[c] = "no_info"

        df = pd.merge(
            df,
            ind[[id_column] + _INDIRECT_COLS],
            on=id_column, how="outer", indicator=True,
        )

        # Si direct_df e indirect_df no tienen los mismos IDs, el outer merge
        # deja NaN. Un NaN NO es una alerta: se rellena como "sin alerta".
        only_direct = df["_merge"] == "left_only"
        only_indirect = df["_merge"] == "right_only"
        if only_direct.any() or only_indirect.any():
            logging.warning(
                f"total_risk: IDs no coinciden entre direct_df e indirect_df "
                f"({int(only_direct.sum()):,} solo en directas, "
                f"{int(only_indirect.sum()):,} solo en indirectas). "
                "Los valores faltantes se rellenan como 'sin alerta'."
            )
        df = df.drop(columns="_merge")

        df["direct_alert"] = df["direct_alert"].map(_str_bool).astype(bool)
        for c in ["deforested_ha", "deforested_prop"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        for c in provided_cols:  # las faltantes conservan "no_info"
            if c in _INDIRECT_BOOL_COLS:
                df[c] = df[c].map(_str_bool).astype(bool)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
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
        df = pd.merge(df, met, on=id_column, how="left")

    # Asegurar columnas de métricas existen (por si no se pasó nada)
    for c in _METRICS_COLS:
        if c not in df.columns:
            df[c] = 0.0

    # -------------------------------------------------------------------------
    # Resultado final
    # -------------------------------------------------------------------------
    elapsed = time.perf_counter() - t0

    if show_progress:
        echo(f"✅ Riesgo total consolidado: {len(df):,} predios en {elapsed:.2f}s")

        # Resumen rápido
        try:
            n_direct = df["direct_alert"].sum()
            n_in = (df["indirect_alert_in"] == True).sum() if "indirect_alert_in" in df.columns else 0  # noqa
            n_out = (df["indirect_alert_out"] == True).sum() if "indirect_alert_out" in df.columns else 0  # noqa
            echo(f"   📊 Alerta directa: {n_direct:,}")
            echo(f"   📊 Alerta indirecta IN: {n_in:,}")
            echo(f"   📊 Alerta indirecta OUT: {n_out:,}")
        except Exception:
            pass

    return df
