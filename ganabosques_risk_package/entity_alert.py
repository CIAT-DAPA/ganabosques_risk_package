# Filename: entity_alert.py
# Description:
#   Compute alerts for different entities based on providers (plots)
#
# Public API:
#   - calculate_alert(alert_indirect_df: pd.DataFrame,entity_df: pd.DataFrame,provider_df: pd.DataFrame,n_workers: int = 2,) -> pd.DataFrame:
#
# Author: Steven Sotelo
#
# Notes:
#   - Uses pandas/numpy/tqdm and optional multiprocessing for assigning results back to the alert_direct table.
#   - Progress is displayed with tqdm over chunked processing of plot IDs.
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from typing import List
from concurrent.futures import ProcessPoolExecutor
import math

import pandas as pd
from tqdm import tqdm


# ----------------------------
# Validation & Normalization
# ----------------------------

def _validate_alert_indirect(df: pd.DataFrame) -> None:
    """Validate required columns in alert_indirect dataframe.

    Required columns:
      - 'id'                : plot identifier
      - 'deforested_area'   : numeric (hectares)
      - 'alert_direct'      : bool
      - 'alert_in'          : bool
      - 'alert_out'         : bool
    """
    required = {"id", "deforested_area", "alert_direct", "alert_in", "alert_out"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"alert_indirect_df is missing required columns: {missing}")


def _normalize_alert_indirect(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with coerced dtypes and safe defaults."""
    out = df.copy()
    out["deforested_area"] = pd.to_numeric(out["deforested_area"], errors="coerce").fillna(0.0)
    for b in ["alert_direct", "alert_in", "alert_out"]:
        out[b] = out[b].fillna(False).astype(bool)
    return out


def _normalize_entity(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize entity master to columns: 'entity_id', 'entity_name'."""
    if not isinstance(entity_df, pd.DataFrame):
        raise ValueError("entity_df must be a pandas DataFrame.")

    df = entity_df.copy()

    # ID normalization
    if "entity_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "entity_id"})
        else:
            raise ValueError("entity_df must contain 'entity_id' or 'id'.")

    # Name normalization
    if "entity_name" not in df.columns:
        #name_candidates = [c for c in ["entity_name", "name", "adm3_name", "adm3"] if c in df.columns]
        name_candidates = [c for c in ["entity_name", "name"] if c in df.columns]
        if not name_candidates:
            raise ValueError("entity_df must contain a name column such as 'entity_name' or 'name'.")
        if "entity_name" not in df.columns:
            df = df.rename(columns={name_candidates[0]: "entity_name"})

    # Keep only needed, unique rows
    return df[["entity_id", "entity_name"]].drop_duplicates().reset_index(drop=True)


def _normalize_provider(provider_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize provider mapping to columns: 'plot_id', 'entity_id'."""
    if not isinstance(provider_df, pd.DataFrame):
        raise ValueError("provider_df must be a pandas DataFrame.")

    df = provider_df.copy()

    # plot_id
    if "plot_id" not in df.columns:
        # Accept a few common variants from upstream
        for c in ["id", "plot", "plotID", "plot_id_alert", "plot_alert_id"]:
            if c in df.columns:
                df = df.rename(columns={c: "plot_id"})
                break
    if "plot_id" not in df.columns:
        raise ValueError("provider_df must contain a plot id column (e.g., 'plot_id').")

    # entity_id
    if "entity_id" not in df.columns:
        for c in ["entity", "adm3_id", "adm2_id", "adm1_id", "id_entity"]:
            if c in df.columns:
                df = df.rename(columns={c: "entity_id"})
                break
    if "entity_id" not in df.columns:
        raise ValueError("provider_df must contain an entity id column (e.g., 'entity_id').")

    # Keep only needed, drop duplicates (a plot can map to multiple entities if many-to-many)
    return df[["plot_id", "entity_id"]].drop_duplicates()


# ----------------------------
# Parallel helpers
# ----------------------------

def _chunk(values: List, n_chunks: int) -> List[List]:
    """Split a list into ~equal contiguous chunks (n_chunks >= 1)."""
    n_chunks = max(1, int(n_chunks))
    L = len(values)
    if L == 0:
        return []
    if n_chunks > L:
        n_chunks = L
    size = math.ceil(L / n_chunks)
    return [values[i:i + size] for i in range(0, L, size)]


def _aggregate_chunk(merged: pd.DataFrame, entity_ids: List) -> pd.DataFrame:
    """Aggregate metrics for a subset of entity_ids.

    merged must contain at least:
      - entity_id
      - id (plot id)
      - alert_direct, alert_in, alert_out (bool)
      - deforested_area (float)
    """
    if not entity_ids:
        return pd.DataFrame(columns=[
            "entity_id",
            "plots_total",
            "plots_alert_direct",
            "plots_alert_in",
            "plots_alert_out",
            "deforested_area_sum",
            "alert",
        ])

    sub = merged[merged["entity_id"].isin(entity_ids)].copy()

    grouped = sub.groupby("entity_id", as_index=False).agg(
        plots_total=("id", "size"),
        plots_alert_direct=("alert_direct", "sum"),
        plots_alert_in=("alert_in", "sum"),
        plots_alert_out=("alert_out", "sum"),
        deforested_area_sum=("deforested_area", "sum"),
        alert=("alert_direct", "max"),  # start with any direct alert
    )

    # Any indirect (in/out) should flip alert to True as well
    grouped["alert"] = (
        grouped["alert"]
        | (grouped["plots_alert_in"] > 0)
        | (grouped["plots_alert_out"] > 0)
    ).astype(bool)

    # Ensure clean dtypes
    grouped["plots_total"] = grouped["plots_total"].astype(int)
    grouped["plots_alert_direct"] = grouped["plots_alert_direct"].astype(int)
    grouped["plots_alert_in"] = grouped["plots_alert_in"].astype(int)
    grouped["plots_alert_out"] = grouped["plots_alert_out"].astype(int)
    grouped["deforested_area_sum"] = grouped["deforested_area_sum"].astype(float)

    return grouped


# ----------------------------
# Public API
# ----------------------------

def calculate_alert(
    alert_indirect_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    provider_df: pd.DataFrame,
    n_workers: int = 2,
) -> pd.DataFrame:
    """Aggregate plot-level alerts to any entity using a provider mapping.

    Args:
        alert_indirect_df:
            DataFrame con las fincas (plots) y sus métricas/flags, generado por `alert_indirect`.
            Debe incluir:
              - 'id'              (plot id)
              - 'deforested_area' (float; ha)
              - 'alert_direct'    (bool)
              - 'alert_in'        (bool)
              - 'alert_out'       (bool)
        entity_df:
            DataFrame maestro de la entidad destino. Se normaliza a:
              - 'entity_id'
              - 'entity_name'
            Se aceptan variantes comunes: ('id' → 'entity_id', 'name' → 'entity_name', etc.).
        provider_df:
            Mapeo entre plots y entidades. Se normaliza a:
              - 'plot_id'
              - 'entity_id'
            Este DF determina qué plots pertenecen a cada entidad.
        n_workers:
            Número de procesos para paralelizar el agregado por entidades. Default: 2.

    Returns:
        DataFrame con una fila por entidad, columnas:
          - entity_id
          - entity_name
          - plots_total
          - plots_alert_direct
          - plots_alert_in
          - plots_alert_out
          - deforested_area_sum
          - alert   (True si existe algún plot con direct OR in OR out)
    """
    # --- Validate & normalize inputs ---
    _validate_alert_indirect(alert_indirect_df)
    plots = _normalize_alert_indirect(alert_indirect_df)
    entity = _normalize_entity(entity_df)
    provider = _normalize_provider(provider_df)

    # --- Build a single merged table: provider × plots (keeps only provider-linked plots) ---
    # Right semantics: we want all provider relations (plots linked to entities)
    merged = provider.merge(
        plots,
        left_on="plot_id",
        right_on="id",
        how="left",
    )

    # If some provider mapping points to plots not present in alert_indirect_df,
    # fill defaults so they contribute safely to counts (non-alert, area 0).
    # (This is conservative; adjust if you prefer dropping those rows.)
    merged["deforested_area"] = pd.to_numeric(merged["deforested_area"], errors="coerce").fillna(0.0)
    for b in ["alert_direct", "alert_in", "alert_out"]:
        merged[b] = merged[b].fillna(False).astype(bool)

    # --- Compute per-entity metrics in parallel ---
    all_entity_ids = entity["entity_id"].drop_duplicates().tolist()
    chunks = _chunk(all_entity_ids, n_chunks=max(1, int(n_workers)))

    partials: List[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as ex:
        futures = [ex.submit(_aggregate_chunk, merged, ch) for ch in chunks]
        for f in tqdm(futures, desc="Aggregating entity alerts", total=len(futures)):
            partials.append(f.result())

    metrics = (
        pd.concat(partials, ignore_index=True)
        if partials
        else pd.DataFrame(
            columns=[
                "entity_id", "plots_total", "plots_alert_direct",
                "plots_alert_in", "plots_alert_out", "deforested_area_sum", "alert"
            ]
        )
    )

    # --- Join metrics back to entity master to keep entities with no plots ---
    out = entity.merge(metrics, on="entity_id", how="left")

    # Fill missing with zeros/False for entities without plots
    out["plots_total"] = out["plots_total"].fillna(0).astype(int)
    out["plots_alert_direct"] = out["plots_alert_direct"].fillna(0).astype(int)
    out["plots_alert_in"] = out["plots_alert_in"].fillna(0).astype(int)
    out["plots_alert_out"] = out["plots_alert_out"].fillna(0).astype(int)
    out["deforested_area_sum"] = out["deforested_area_sum"].fillna(0.0).astype(float)
    out["alert"] = out["alert"].fillna(False).astype(bool)

    # Stable column order
    out = out[
        [
            "entity_id",
            "entity_name",
            "plots_total",
            "plots_alert_direct",
            "plots_alert_in",
            "plots_alert_out",
            "deforested_area_sum",
            "alert",
        ]
    ].sort_values(by="entity_id").reset_index(drop=True)

    return out


# ----------------------------
# Example usage (commented)
# ----------------------------
# if __name__ == "__main__":
#     # alert_indirect_df: produced by your alert_indirect pipeline
#     alert_indirect_df = pd.DataFrame({
#         "id": [101, 102, 103, 104, 105],
#         "deforested_area": [0.5, 1.2, 0.0, 3.3, 0.0],
#         "alert_direct": [True, False, False, True, False],
#         "alert_in": [False, True, False, False, False],
#         "alert_out": [False, False, True, False, False],
#     })
#
#     # entity_df: any master table (e.g., ADM3/association/co-op/etc.)
#     entity_df = pd.DataFrame({
#         "id": ["X", "Y", "Z", "W"],
#         "name": ["Alpha", "Beta", "Gamma", "NoPlots"]
#     })
#
#     # provider_df: mapping plot -> entity
#     provider_df = pd.DataFrame({
#         "plot_id": [101, 102, 103, 104, 105, 999],  # 999 not present in plots
#         "entity_id": ["X",  "X",  "Y",  "Y",  "Z",  "Z"],
#     })
#
#     out = calculate_alert(alert_indirect_df, entity_df, provider_df, n_workers=2)
#     print(out)
