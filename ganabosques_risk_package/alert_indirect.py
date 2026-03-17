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

    # -------------------------------------------------------------------------
    # Preparar datos
    # -------------------------------------------------------------------------
    # Construir diccionario id -> tiene_alerta
    alert_ids_bool = {
        str(row[id_column]): _str_bool(row["direct_alert"])
        for _, row in alert_direct_df.iterrows()
    }

    # Normalizar IDs en movimientos a string
    mov = movements_df.copy()
    mov["origen_id"] = mov["origen_id"].astype(str)
    mov["destination_id"] = mov["destination_id"].astype(str)

    # Diagnóstico
    ids_alerta = set(alert_ids_bool.keys())
    origenes = set(mov["origen_id"].unique())
    destinos = set(mov["destination_id"].unique())
    universo_mov = origenes | destinos
    inter = ids_alerta & universo_mov

    n_alertados = sum(alert_ids_bool.values())
    n_total = len(alert_ids_bool)
    n_mov = len(mov)

    if show_progress:
        print(f"🔄 Calculando alertas indirectas:")
        print(f"   • Predios: {n_total:,} ({n_alertados:,} con alerta directa)")
        print(f"   • Movimientos: {n_mov:,}")
        print(f"   • Cruce IDs (alertas ∩ movimientos): {len(inter):,}")

    if len(inter) == 0:
        logging.warning(
            "No hay cruce de IDs entre alertas y movimientos. "
            "Verifica que los IDs estén normalizados."
        )

    # -------------------------------------------------------------------------
    # Marcar movimientos con alerta en origen/destino
    # -------------------------------------------------------------------------
    mov["origin_has_alert"] = mov["origen_id"].map(
        lambda k: bool(alert_ids_bool.get(k, False))
    )
    mov["dest_has_alert"] = mov["destination_id"].map(
        lambda k: bool(alert_ids_bool.get(k, False))
    )

    # -------------------------------------------------------------------------
    # Calcular métricas
    # -------------------------------------------------------------------------
    ids = pd.Index(sorted(alert_ids_bool.keys()))

    # Total de movimientos entrantes/salientes por predio
    n_in = mov.groupby("destination_id").size().reindex(ids).fillna(0).astype(int)
    n_out = mov.groupby("origen_id").size().reindex(ids).fillna(0).astype(int)

    # Movimientos indirectos (desde/hacia predios con alerta directa)
    tmp_in = mov[mov["origin_has_alert"]].groupby("destination_id").size()
    tmp_out = mov[mov["dest_has_alert"]].groupby("origen_id").size()

    result = pd.DataFrame({
        "id": ids,
        "n_in": n_in.values,
        "n_out": n_out.values,
        "n_indirect_in": tmp_in.reindex(ids).fillna(0).astype(int).values,
        "n_indirect_out": tmp_out.reindex(ids).fillna(0).astype(int).values,
    })

    result["n_total_mov"] = result["n_in"] + result["n_out"]
    result["indirect_alert_in"] = result["n_indirect_in"] > 0
    result["indirect_alert_out"] = result["n_indirect_out"] > 0

    elapsed = time.perf_counter() - t0

    # Estadísticas
    n_alert_in = result["indirect_alert_in"].sum()
    n_alert_out = result["indirect_alert_out"].sum()
    if show_progress:
        print(f"✅ Alertas indirectas: {n_total:,} predios en {elapsed:.2f}s")
        print(f"   📊 Con alerta IN (recibe de alertado): {n_alert_in:,}")
        print(f"   📊 Con alerta OUT (envía a alertado): {n_alert_out:,}")

    return result
