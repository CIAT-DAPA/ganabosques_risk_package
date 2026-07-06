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
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _str_bool(x) -> bool:
    """Convierte valor a booleano, tolerando strings 'True'/'False'."""
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("true", "1", "yes", "si", "sí")
    try:
        return bool(x)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def alert_indirect(
    alert_direct_df: pd.DataFrame,
    movements_df: pd.DataFrame,
    id_column: str = "id",
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
    show_progress : bool, default True
        Mostrar información de progreso.

    Returns
    -------
    pandas.DataFrame
        Una fila por predio (mismo universo que alert_direct_df) con:
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
    alert_df = alert_direct_df[[id_column, "direct_alert"]].copy()
    alert_df[id_column] = alert_df[id_column].astype(str)

    alert_series = (
        alert_df
        .set_index(id_column)["direct_alert"]
        .map(_str_bool)
    )

    mov = movements_df.copy()
    mov["origen_id"] = mov["origen_id"].astype(str)
    mov["destination_id"] = mov["destination_id"].astype(str)

    # ---------------------------------------------------------------------
    # Diagnóstico
    # ---------------------------------------------------------------------
    ids_alerta = set(alert_series.index)
    origenes = set(mov["origen_id"].unique())
    destinos = set(mov["destination_id"].unique())
    universo_mov = origenes | destinos
    inter = ids_alerta & universo_mov

    if show_progress:
        print(f"🔄 Calculando alertas indirectas:")
        print(f"   • Predios: {len(ids_alerta):,}")
        print(f"   • Movimientos: {len(mov):,}")
        print(f"   • Cruce IDs: {len(inter):,}")

    if len(inter) == 0:
        logging.warning(
            "No hay cruce de IDs entre alertas y movimientos."
        )

    # ---------------------------------------------------------------------
    # Flags de alerta 
    # ---------------------------------------------------------------------
    mov["origin_has_alert"] = mov["origen_id"].map(alert_series).fillna(False).astype(bool)
    mov["dest_has_alert"] = mov["destination_id"].map(alert_series).fillna(False).astype(bool)

    # ---------------------------------------------------------------------
    # Universo de IDs a reportar (todos los de alert_direct_df)
    # ---------------------------------------------------------------------
    ids = pd.Index(alert_series.index)

    # ---------------------------------------------------------------------
    # Agregaciones 
    # ---------------------------------------------------------------------
    n_in = mov.groupby("destination_id").size().reindex(ids, fill_value=0)
    n_out = mov.groupby("origen_id").size().reindex(ids, fill_value=0)

    n_indirect_in = (
        mov.loc[mov["origin_has_alert"]]
        .groupby("destination_id")
        .size()
        .reindex(ids, fill_value=0)
    )

    n_indirect_out = (
        mov.loc[mov["dest_has_alert"]]
        .groupby("origen_id")
        .size()
        .reindex(ids, fill_value=0)
    )

    # ---------------------------------------------------------------------
    # Resultado final
    # ---------------------------------------------------------------------
    result = pd.DataFrame({
        "id": ids,
        "n_in": n_in,
        "n_out": n_out,
        "n_indirect_in": n_indirect_in,
        "n_indirect_out": n_indirect_out,
    }).reset_index(drop=True)

    result["n_total_mov"] = result["n_in"] + result["n_out"]
    result["indirect_alert_in"] = result["n_indirect_in"] > 0
    result["indirect_alert_out"] = result["n_indirect_out"] > 0

    elapsed = time.perf_counter() - t0

    if show_progress:
        print(f"✅ Listo en {elapsed:.2f}s")
        print(f"   📊 IN: {result['indirect_alert_in'].sum():,}")
        print(f"   📊 OUT: {result['indirect_alert_out'].sum():,}")

    return result