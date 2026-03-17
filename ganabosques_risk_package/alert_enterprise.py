# -*- coding: utf-8 -*-
"""
Alertas de empresa basadas en movimientos de ganado.

Identifica empresas (mataderos, ferias, procesadores, etc.) que tuvieron
movimientos con fincas que presentaron alguna alerta (directa o indirecta).

Recibe DataFrames puros.

Public API:
  - alert_enterprise(total_risk_df, movements_df, ...) -> pd.DataFrame

"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_ENTERPRISE_TYPES = [
    "SLAUGHTERHOUSE",
    "CATTLE_FAIR",
    "PROCESSOR",
    "ENTERPRISE",
]


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------

def _normalize_id(value) -> str:
    """Normaliza un ID a string limpio (mayúsculas, sin ceros a la izq.)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if s.lower() == "nan":
        return ""
    # Quitar .0 de números flotantes
    if re.match(r"^\d+\.0$", s):
        s = s[:-2]
    # Quitar ceros a la izquierda
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
# Public API
# ---------------------------------------------------------------------------

def alert_enterprise(
    total_risk_df: pd.DataFrame,
    movements_df: pd.DataFrame,
    id_column: str = "id",
    enterprise_types: Optional[List[str]] = None,
    alert_columns: Optional[List[str]] = None,
    normalize_ids: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Identifica empresas conectadas con fincas con alerta vía movimientos.

    Lógica:
      1. Determina las fincas con *cualquier* alerta (directa o indirecta).
      2. Filtra movimientos donde **origen** es finca alertada y **destino** es
         tipo empresa → ``typemove = "in"`` (la finca envía a la empresa).
      3. Filtra movimientos donde **origen** es tipo empresa y **destino** es
         finca alertada → ``typemove = "out"`` (la empresa envía a finca alertada).
      4. Combina, deduplica, y devuelve el DataFrame resultante.

    Parameters
    ----------
    total_risk_df : pandas.DataFrame
        Resultado de ``total_risk()`` (o de ``alert_direct()``). Debe tener al
        menos la columna ``id_column`` y al menos una columna de alerta.
    movements_df : pandas.DataFrame
        DataFrame de movimientos. Columnas esperadas:

        - ``origen_id`` : ID de origen del movimiento (finca o empresa).
        - ``destination_id`` : ID de destino del movimiento.
        - ``tipo_origen`` : Tipo del origen (``"FARM"``, ``"SLAUGHTERHOUSE"``, etc.).
        - ``tipo_destino`` : Tipo del destino.
        - ``producer_id_origen`` (opcional) : Código del productor origen.
        - ``producer_id_destino`` (opcional) : Código del productor destino.

        Las columnas pueden tener nombres en mayúsculas (se renombran).
    id_column : str, default "id"
        Columna de ID en ``total_risk_df``.
    enterprise_types : list of str, optional
        Tipos de empresa a considerar.
        Default: ``["SLAUGHTERHOUSE", "CATTLE_FAIR", "PROCESSOR", "ENTERPRISE"]``.
    alert_columns : list of str, optional
        Columnas booleanas que definen si una finca tiene alerta.
        Default: ``["direct_alert", "indirect_alert_in", "indirect_alert_out"]``.
        Una finca se considera alertada si *cualquiera* de estas es True.
    normalize_ids : bool, default True
        Si True, normaliza IDs (quitar ceros, mayúsculas, etc.).
    show_progress : bool, default True
        Mostrar resumen.

    Returns
    -------
    pandas.DataFrame
        Columnas: ``idpro, id_farm, typemove, enterprise_type,
        farm_has_direct_alert, farm_has_indirect_alert``.
        Una fila por combinación única (idpro, id_farm, typemove).

    Examples
    --------
    >>> risk = total_risk(direct_df, indirect_df)
    >>> ent = alert_enterprise(risk, movements)
    >>> ent[ent["typemove"] == "in"]  # fincas alertadas que envían a empresas
    """
    t0 = time.perf_counter()

    if enterprise_types is None:
        enterprise_types = list(DEFAULT_ENTERPRISE_TYPES)

    if alert_columns is None:
        alert_columns = ["direct_alert", "indirect_alert_in", "indirect_alert_out"]

    # ------------------------------------------------------------------
    # 1. Determinar fincas con alerta
    # ------------------------------------------------------------------
    risk = total_risk_df.copy()
    risk[id_column] = risk[id_column].astype(str)
    if normalize_ids:
        risk[id_column] = risk[id_column].apply(_normalize_id)

    # Evaluar cuáles tienen alerta
    existing_alert_cols = [c for c in alert_columns if c in risk.columns]
    if not existing_alert_cols:
        logging.warning("No se encontraron columnas de alerta en total_risk_df. "
                        f"Buscadas: {alert_columns}")
        return pd.DataFrame(columns=[
            "idpro", "id_farm", "typemove", "enterprise_type",
            "farm_has_direct_alert", "farm_has_indirect_alert",
        ])

    # Finca alertada = cualquier columna de alerta es True
    risk["_has_any_alert"] = False
    for col in existing_alert_cols:
        risk["_has_any_alert"] = risk["_has_any_alert"] | risk[col].map(_str_bool)

    farms_with_alert = set(risk.loc[risk["_has_any_alert"], id_column].unique())

    if not farms_with_alert:
        logging.warning("No hay fincas con alerta.")
        return pd.DataFrame(columns=[
            "idpro", "id_farm", "typemove", "enterprise_type",
            "farm_has_direct_alert", "farm_has_indirect_alert",
        ])

    # Útil para enriquecer después
    farm_direct_alert = set(
        risk.loc[risk.get("direct_alert", pd.Series(dtype=bool)).map(_str_bool),
                 id_column].unique()
    ) if "direct_alert" in risk.columns else set()

    farm_indirect_alert = set()
    for ic in ["indirect_alert_in", "indirect_alert_out"]:
        if ic in risk.columns:
            farm_indirect_alert |= set(
                risk.loc[risk[ic].map(_str_bool), id_column].unique()
            )

    if show_progress:
        print(f"🏭 Buscando alertas de empresa para {len(farms_with_alert):,} "
              f"fincas alertadas en {len(movements_df):,} movimientos")

    # ------------------------------------------------------------------
    # 2. Preparar movimientos
    # ------------------------------------------------------------------
    mov = movements_df.copy()

    # Renombrar columnas estándar (mayúsculas → minúsculas)
    col_map = {
        "SIT_CODE_ORIGEN": "origen_id",
        "SIT_CODE_DESTINO": "destination_id",
        "TIPO_ORIGEN": "tipo_origen",
        "TIPO_DESTINO": "tipo_destino",
        "PRODUCER_ID_ORIGEN": "producer_id_origen",
        "PRODUCER_ID_DESTINO": "producer_id_destino",
        "IDPRO": "idpro",
        "ID_PRO": "idpro",
    }
    mov = mov.rename(columns={k: v for k, v in col_map.items() if k in mov.columns})

    # Verificar columnas mínimas
    for req in ("origen_id", "destination_id"):
        if req not in mov.columns:
            raise ValueError(f"movements_df debe contener columna '{req}'.")

    # Normalizar IDs
    if normalize_ids:
        mov["origen_id"] = mov["origen_id"].astype(str).apply(_normalize_id)
        mov["destination_id"] = mov["destination_id"].astype(str).apply(_normalize_id)
        for pcol in ("producer_id_origen", "producer_id_destino"):
            if pcol in mov.columns:
                mov[pcol] = mov[pcol].astype(str).apply(_normalize_id)

    # Normalizar tipos
    for tcol in ("tipo_origen", "tipo_destino"):
        if tcol in mov.columns:
            mov[tcol] = mov[tcol].astype(str).str.strip().str.upper()

    # ------------------------------------------------------------------
    # 3. Finca → Empresa  (typemove = "in")
    #    origen = finca alertada, destino = tipo empresa
    # ------------------------------------------------------------------
    has_tipo_destino = "tipo_destino" in mov.columns
    if has_tipo_destino:
        mask_to_ent = (
            mov["origen_id"].isin(farms_with_alert) &
            mov["tipo_destino"].isin(enterprise_types)
        )
    else:
        # Sin tipo, tomar todos los destinos que NO sean fincas (heurística)
        mask_to_ent = mov["origen_id"].isin(farms_with_alert)

    entries = mov[mask_to_ent].copy()
    entries["typemove"] = "in"
    entries["id_farm"] = entries["origen_id"]
    if "producer_id_destino" in entries.columns:
        entries["idpro"] = entries["producer_id_destino"]
    elif "idpro" not in entries.columns:
        entries["idpro"] = entries["destination_id"]
    if has_tipo_destino:
        entries["enterprise_type"] = entries["tipo_destino"]
    else:
        entries["enterprise_type"] = ""

    # ------------------------------------------------------------------
    # 4. Empresa → Finca  (typemove = "out")
    #    origen = tipo empresa, destino = finca alertada
    # ------------------------------------------------------------------
    has_tipo_origen = "tipo_origen" in mov.columns
    if has_tipo_origen:
        mask_from_ent = (
            mov["destination_id"].isin(farms_with_alert) &
            mov["tipo_origen"].isin(enterprise_types)
        )
    else:
        mask_from_ent = mov["destination_id"].isin(farms_with_alert)

    exits_ = mov[mask_from_ent].copy()
    exits_["typemove"] = "out"
    exits_["id_farm"] = exits_["destination_id"]
    if "producer_id_origen" in exits_.columns:
        exits_["idpro"] = exits_["producer_id_origen"]
    elif "idpro" not in exits_.columns:
        exits_["idpro"] = exits_["origen_id"]
    if has_tipo_origen:
        exits_["enterprise_type"] = exits_["tipo_origen"]
    else:
        exits_["enterprise_type"] = ""

    # ------------------------------------------------------------------
    # 5. Combinar & deduplicar
    # ------------------------------------------------------------------
    result = pd.concat([entries, exits_], ignore_index=True)

    if result.empty:
        if show_progress:
            print("⚠️  No se encontraron movimientos con fincas alertadas.")
        return pd.DataFrame(columns=[
            "idpro", "id_farm", "typemove", "enterprise_type",
            "farm_has_direct_alert", "farm_has_indirect_alert",
        ])

    # Enriquecer con flags de tipo de alerta
    result["farm_has_direct_alert"] = result["id_farm"].isin(farm_direct_alert)
    result["farm_has_indirect_alert"] = result["id_farm"].isin(farm_indirect_alert)

    # Seleccionar columnas y deduplicar
    out_cols = [
        "idpro", "id_farm", "typemove", "enterprise_type",
        "farm_has_direct_alert", "farm_has_indirect_alert",
    ]
    out_cols = [c for c in out_cols if c in result.columns]
    result = result[out_cols].drop_duplicates(subset=["idpro", "id_farm", "typemove"])

    elapsed = time.perf_counter() - t0

    if show_progress:
        n_in = (result["typemove"] == "in").sum()
        n_out = (result["typemove"] == "out").sum()
        n_ent = result["idpro"].nunique()
        n_farm = result["id_farm"].nunique()
        print(f"✅ Alertas de empresa: {len(result):,} registros en {elapsed:.2f}s")
        print(f"   📊 Movimientos IN (finca→empresa): {n_in:,}")
        print(f"   📊 Movimientos OUT (empresa→finca): {n_out:,}")
        print(f"   📊 Empresas únicas: {n_ent:,}")
        print(f"   📊 Fincas únicas involucradas: {n_farm:,}")

    return result.reset_index(drop=True)
