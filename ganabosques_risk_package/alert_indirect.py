# -*- coding: utf-8 -*-
"""
Cálculo de alertas indirectas de deforestación por movimiento.

A partir de las alertas directas y una tabla de movimientos entre predios,
determina para cada predio si recibe de o envía hacia predios alertados.

Salida por predio:
  - n_in / n_out: total de movimientos entrantes/salientes
  - n_indirect_in / n_indirect_out: movimientos desde/hacia predios alertados
  - n_total_mov: total de movimientos
  - indirect_alert_in / indirect_alert_out: booleanos

Public API:
  - alert_indirect(alert_direct_df, movements_df) -> pd.DataFrame

"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from ._utils import echo, normalize_id_series, str_bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def alert_indirect(
    alert_direct_df: pd.DataFrame,
    movements_df: pd.DataFrame,
    id_column: str = "id",
    normalize_ids: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Calcula alertas indirectas basadas en movimientos entre predios.

    Para cada predio en ``alert_direct_df``, determina si tiene movimientos
    entrantes o salientes desde/hacia predios con alerta directa.

    Parameters
    ----------
    alert_direct_df : pandas.DataFrame
        Resultado de ``alert_direct()`` con al menos:
          - columna ``id_column`` (identificador del predio)
          - ``direct_alert`` (bool)
    movements_df : pandas.DataFrame
        Movimientos entre predios con al menos:
          - ``origen_id``: ID del predio de origen
          - ``destination_id``: ID del predio de destino
        Puede contener columnas adicionales (date, tipo, etc.) que se ignoran.
    id_column : str, default "id"
        Nombre de la columna de ID en ``alert_direct_df``.
    normalize_ids : bool, default True
        Si True, los IDs se normalizan **solo para el cruce** (``1``, ``1.0``,
        ``"0001"`` se consideran el mismo predio), igual que en
        ``alert_enterprise`` y ``supplier_risk``. La columna ``id`` de salida
        conserva el valor original (como string), para que siga empatando con
        la salida de ``alert_direct`` en ``total_risk``.
    show_progress : bool, default True
        Mostrar información de progreso.

    Returns
    -------
    pandas.DataFrame
        Una fila por ID único de alert_direct_df (si un ID aparece varias
        veces, se considera alertado si cualquiera de sus filas lo está) con:
          - id: identificador del predio
          - n_in: total de movimientos entrantes
          - n_out: total de movimientos salientes
          - n_indirect_in: movimientos entrantes desde predios con alerta
          - n_indirect_out: movimientos salientes hacia predios con alerta
          - n_total_mov: n_in + n_out
          - indirect_alert_in: bool (n_indirect_in > 0)
          - indirect_alert_out: bool (n_indirect_out > 0)

    Raises
    ------
    ValueError
        Si faltan columnas requeridas.

    Examples
    --------
    >>> direct = alert_direct(plots, raster)
    >>> movements = pd.read_csv("movements.csv")
    >>> indirect = alert_indirect(direct, movements)
    """
    
    # -------------------------------------------------------------------------
    # Validaciones
    # -------------------------------------------------------------------------
    if id_column not in alert_direct_df.columns:
        raise ValueError(
            f"alert_direct_df debe contener columna '{id_column}'. "
            f"Columnas: {list(alert_direct_df.columns)}"
        )

    if "direct_alert" not in alert_direct_df.columns:
        raise ValueError(
            "alert_direct_df debe contener columna 'direct_alert'."
        )

    required_mov = {"origen_id", "destination_id"}
    missing = required_mov - set(movements_df.columns)
    if missing:
        raise ValueError(
            f"movements_df requiere columnas: {sorted(missing)}. "
            f"Columnas: {list(movements_df.columns)}"
        )

    t0 = time.perf_counter()

    # ---------------------------------------------------------------------
    # Preparación (vectorizada)
    # ---------------------------------------------------------------------
    def _keys(series: pd.Series) -> pd.Series:
        if normalize_ids:
            return normalize_id_series(series)
        return series.astype(str)

    alert_df = alert_direct_df[[id_column, "direct_alert"]].copy()
    alert_df["_orig_id"] = alert_df[id_column].astype(str)
    alert_df["_key"] = _keys(alert_df[id_column])
    alert_df["_alert"] = alert_df["direct_alert"].map(str_bool).astype(bool)

    n_dup = int(alert_df["_orig_id"].duplicated().sum())
    if n_dup:
        logging.warning(
            f"alert_direct_df tiene {n_dup:,} IDs duplicados; se reporta una fila "
            "por ID (alertado si cualquiera de sus filas lo está)."
        )

    # Alerta por clave normalizada (any() por si hay duplicados). La clave
    # vacía corresponde a IDs faltantes y nunca debe cruzar con movimientos.
    alert_series = alert_df.groupby("_key", sort=False)["_alert"].any()
    alert_series = alert_series[alert_series.index != ""]

    mov = movements_df[["origen_id", "destination_id"]].copy()
    mov["_o"] = _keys(mov["origen_id"])
    mov["_d"] = _keys(mov["destination_id"])

    # ---------------------------------------------------------------------
    # Diagnóstico
    # ---------------------------------------------------------------------
    ids_alerta = set(alert_series.index)
    universo_mov = set(mov["_o"].unique()) | set(mov["_d"].unique())
    inter = ids_alerta & universo_mov

    if show_progress:
        echo(f"🔄 Calculando alertas indirectas:")
        echo(f"   • Predios: {len(ids_alerta):,}")
        echo(f"   • Movimientos: {len(mov):,}")
        echo(f"   • Cruce IDs: {len(inter):,}")

    if len(inter) == 0:
        logging.warning(
            "No hay cruce de IDs entre alertas y movimientos."
        )

    # ---------------------------------------------------------------------
    # Flags de alerta (claves sin predio conocido -> False)
    # ---------------------------------------------------------------------
    mov["origin_has_alert"] = mov["_o"].map(alert_series).eq(True)
    mov["dest_has_alert"] = mov["_d"].map(alert_series).eq(True)

    # ---------------------------------------------------------------------
    # Agregaciones por clave
    # ---------------------------------------------------------------------
    keys = pd.Index(alert_series.index)

    per_key = pd.DataFrame({
        "n_in": mov.groupby("_d").size().reindex(keys, fill_value=0),
        "n_out": mov.groupby("_o").size().reindex(keys, fill_value=0),
        "n_indirect_in": (
            mov.loc[mov["origin_has_alert"]].groupby("_d").size()
            .reindex(keys, fill_value=0)
        ),
        "n_indirect_out": (
            mov.loc[mov["dest_has_alert"]].groupby("_o").size()
            .reindex(keys, fill_value=0)
        ),
    }, index=keys)

    # ---------------------------------------------------------------------
    # Resultado final: una fila por ID original (orden de alert_direct_df)
    # ---------------------------------------------------------------------
    count_cols = ["n_in", "n_out", "n_indirect_in", "n_indirect_out"]
    result = (
        alert_df[["_orig_id", "_key"]]
        .drop_duplicates("_orig_id")
        .join(per_key, on="_key")
        .drop(columns="_key")
        .rename(columns={"_orig_id": "id"})
        .reset_index(drop=True)
    )
    result[count_cols] = result[count_cols].fillna(0).astype(int)

    result["n_total_mov"] = result["n_in"] + result["n_out"]
    result["indirect_alert_in"] = result["n_indirect_in"] > 0
    result["indirect_alert_out"] = result["n_indirect_out"] > 0

    elapsed = time.perf_counter() - t0

    if show_progress:
        echo(f"✅ Listo en {elapsed:.2f}s")
        echo(f"   📊 IN: {result['indirect_alert_in'].sum():,}")
        echo(f"   📊 OUT: {result['indirect_alert_out'].sum():,}")

    return result