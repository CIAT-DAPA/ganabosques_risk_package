# Filename: plot_alert_indirect.py
# Description:
#   Compute indirect alerts for plots based on movement between plots and the
#   direct-alert status of origins/destinations.
#
# Public API:
#   - alert_indirect(alert_direct_df: pd.DataFrame, movement_df: pd.DataFrame, n_workers: int = 2) -> pd.DataFrame
#
# Author: Steven Sotelo
#
# Notes:
#   - Uses pandas/numpy/tqdm and optional multiprocessing for assigning results back to the alert_direct table.
#   - Progress is displayed with tqdm over chunked processing of plot IDs.
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from typing import Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor
import math

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Internal helpers
# ----------------------------

def _validate_inputs(alert_direct_df: pd.DataFrame, movement_df: pd.DataFrame) -> None:
    """Validate minimally required columns in inputs.

    Args:
        alert_direct_df: DataFrame produced by `alert_direct(...)`. Must contain:
            - 'id' (plot identifier)
            - 'alert_direct' (boolean flag)
        movement_df: DataFrame with movements. Must contain:
            - 'origen_id'
            - 'destination_id'

    Raises:
        ValueError: if required columns are missing.
    """
    required_ad = {"id", "alert_direct"}
    required_mv = {"origen_id", "destination_id"}

    missing_ad = required_ad - set(alert_direct_df.columns)
    missing_mv = required_mv - set(movement_df.columns)

    if missing_ad:
        raise ValueError(f"alert_direct_df is missing required columns: {sorted(missing_ad)}")
    if missing_mv:
        raise ValueError(f"movement_df is missing required columns: {sorted(missing_mv)}")


def _build_new_dataframe(
    alert_direct_df: pd.DataFrame,
    movement_df: pd.DataFrame,
) -> pd.DataFrame:
    """Construct the consolidated DataFrame with both origin and destination annotations.

    Produces boolean columns:
      - alert_direct_origen
      - alert_direct_destination
    """
    # Work on a copy with a clean incremental index
    mv = movement_df.reset_index(drop=True).copy()

    # Keep only the minimal columns we need from alert_direct_df to avoid name clutter
    ad_min = alert_direct_df[["id", "alert_direct"]].copy()

    # ---- ORIGIN side: merge on (origen_id == id) and rename alert flag
    mv = mv.merge(
        ad_min,
        how="left",
        left_on="origen_id",
        right_on="id",
    )
    # Rename and drop the join helper column 'id' that came from the right
    mv = mv.rename(columns={"alert_direct": "alert_direct_origen"})
    mv = mv.drop(columns=["id"])

    # ---- DESTINATION side: merge on (destination_id == id) and rename alert flag
    mv = mv.merge(
        ad_min,
        how="left",
        left_on="destination_id",
        right_on="id",
    )
    mv = mv.rename(columns={"alert_direct": "alert_direct_destination"})
    mv = mv.drop(columns=["id"])

    # Ensure pure booleans with safe defaults (NaN -> False)
    mv["alert_direct_origen"] = mv["alert_direct_origen"].fillna(False).astype(bool)
    mv["alert_direct_destination"] = mv["alert_direct_destination"].fillna(False).astype(bool)

    return mv


def _compute_lookup_dicts(new_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """From the consolidated movement table, build two boolean lookup dictionaries.

    Logic:
      - For each `origen_id`, does it have ANY movement whose `destination` has `alert_direct=True`?
        => origin_to_has_dest_alert[origen_id] = any(destination_alert_direct)
      - For each `destination_id`, does it have ANY movement whose `origin` has `alert_direct=True`?
        => dest_to_has_origin_alert[destination_id] = any(origen_alert_direct)

    Args:
        new_df: Consolidated movement dataframe with both side annotations.

    Returns:
        (origin_to_has_dest_alert, dest_to_has_origin_alert):
            Two dicts mapping plot_id -> bool.
    """
    # Group and reduce with .any() (vectorized, fast).
    origin_to_has_dest_alert = (
        new_df.groupby("origen_id")["alert_direct_destination"].any().to_dict()
        if "alert_direct_destination" in new_df.columns else {}
    )
    dest_to_has_origin_alert = (
        new_df.groupby("destination_id")["alert_direct_origen"].any().to_dict()
        if "alert_direct_origen" in new_df.columns else {}
    )
    return origin_to_has_dest_alert, dest_to_has_origin_alert


def _assign_chunk(ids: List, origin_dict: Dict, dest_dict: Dict) -> pd.DataFrame:
    """Worker function to assign alert_in/alert_out for a chunk of plot ids.

    Args:
        ids: list of plot IDs (values from alert_direct_df['id']).
        origin_dict: dict mapping plot_id -> bool for alert_in logic
                     (does this id as ORIGIN reach any destination with direct alert?).
        dest_dict: dict mapping plot_id -> bool for alert_out logic
                   (does this id as DESTINATION receive from any origin with direct alert?).

    Returns:
        DataFrame with columns ['id', 'alert_in', 'alert_out'] for the given ids.
    """
    out = pd.DataFrame({"id": ids})
    # alert_in: lookups by id in origin_dict (origin_id == id & any destination_alert_direct)
    out["alert_in"] = [bool(origin_dict.get(i, False)) for i in ids]
    # alert_out: lookups by id in dest_dict (destination_id == id & any origen_alert_direct)
    out["alert_out"] = [bool(dest_dict.get(i, False)) for i in ids]
    return out


def _chunked(lst: List, n: int) -> List[List]:
    """Split a list into n approximately equal contiguous chunks (n >= 1)."""
    n = max(1, int(n))
    L = len(lst)
    if n > L:
        n = L
    size = math.ceil(L / n)
    return [lst[i:i + size] for i in range(0, L, size)]


# ----------------------------
# Public API
# ----------------------------

def alert_indirect(
    alert_direct_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    n_workers: int = 2,
) -> pd.DataFrame:
    """Compute indirect alerts (`alert_in`, `alert_out`) from movement graph and direct alerts.

    Detailed steps (mirrors your specification):

    1) **Joins (twice)**:
       - Join `alert_direct_df.id` with `movement_df.origen_id` (semantics of a *right join* on movement):
         the resulting columns from `alert_direct_df` are **prefixed with `origen_`**.
       - Join `alert_direct_df.id` with `movement_df.destination_id` (right join):
         the resulting columns from `alert_direct_df` are **prefixed with `destination_`**.
       We consolidate both into a single `new_df` aligned with `movement_df`.

    2) **Per-plot indirect alerts** (computed back on `alert_direct_df`):
       - For each plot `id`, **alert_in** is `True` if there exists *any* row in `new_df`
         where `origen_id == id` **and** `destination_alert_direct == True`.
       - For each plot `id`, **alert_out** is `True` if there exists *any* row in `new_df`
         where `destination_id == id` **and** `origen_alert_direct == True`.

    3) **Parallel assignment with progress**:
       - We build two lookup dictionaries via vectorized groupbys.
       - We then split plot ids into chunks and compute `alert_in`/`alert_out` per-chunk
         in parallel using `ProcessPoolExecutor` (default `n_workers=2`), reporting progress with `tqdm`.

    Args:
        alert_direct_df: DataFrame from `alert_direct(...)` with at least:
            - 'id' (unique plot identifier)
            - 'alert_direct' (bool)
            (It may include additional columns like areas, proportions, etc.)
        movement_df: DataFrame that describes movements between plots with:
            - 'origen_id' (origin plot id)
            - 'destination_id' (destination plot id)
            (Optional extra metadata columns are preserved.)
        n_workers: Number of worker processes for parallel assignment. Default: 2.

    Returns:
        pd.DataFrame: A **copy** of `alert_direct_df` augmented with:
            - 'alert_in'  (bool)
            - 'alert_out' (bool)

        Additionally, for debugging/inspection you may want to also return `new_df`.
        If you need that, we can add an optional flag to return both.
    """
    _validate_inputs(alert_direct_df, movement_df)

    # 1) Build consolidated movement-enriched table with both origin and destination annotations.
    new_df = _build_new_dataframe(alert_direct_df, movement_df)

    # 2) Vectorized lookups (fast).
    origin_dict, dest_dict = _compute_lookup_dicts(new_df)

    # IDs to compute for
    plot_ids: List = alert_direct_df["id"].tolist()

    # 3) Parallel assignment over chunks with progress bar.
    chunks = _chunked(plot_ids, n_workers)
    partial_results: List[pd.DataFrame] = []

    with ProcessPoolExecutor(max_workers=max(1, int(n_workers))) as ex:
        futures = [ex.submit(_assign_chunk, ch, origin_dict, dest_dict) for ch in chunks]
        for f in tqdm(futures, desc="Computing indirect alerts", total=len(futures)):
            partial_results.append(f.result())

    assigned = pd.concat(partial_results, ignore_index=True)

    # Merge assigned flags back into a copy of alert_direct_df
    out_df = alert_direct_df.copy()
    out_df = out_df.merge(assigned, how="left", on="id")

    # Ensure booleans (fill NaN -> False if no movements)
    for col in ["alert_in", "alert_out"]:
        if col in out_df.columns:
            out_df[col] = out_df[col].fillna(False).astype(bool)
        else:
            out_df[col] = False

    return out_df


# ----------------------------
# Example usage (commented)
# ----------------------------
# if __name__ == "__main__":
#     # Suppose you have:
#     #   alert_direct_df = pd.read_parquet("alert_direct.parquet")  # must include 'id' and 'alert_direct'
#     #   movement_df = pd.read_parquet("movement.parquet")          # must include 'origen_id','destination_id'
#     #
#     # Quick demo with tiny synthetic data:
#     ad = pd.DataFrame({
#         "id": ["A", "B", "C"],
#         "alert_direct": [True, False, False],
#         "plot_area": [1.2, 2.5, 3.3],  # extra fields are preserved
#     })
#     mv = pd.DataFrame({
#         "origen_id": ["A", "B", "C", "B"],
#         "destination_id": ["B", "C", "A", "A"],
#         "weight": [1, 1, 1, 1],  # optional metadata
#     })
#     out = alert_indirect(ad, mv, n_workers=2)
#     print(out)
