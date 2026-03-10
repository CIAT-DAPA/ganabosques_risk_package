# -*- coding: utf-8 -*-
"""
Cálculo de métricas espaciales para predios (plots/farms).

Calcula para cada predio:
  - total_ha: Área total del predio en hectáreas
  - farming_in_ha / farming_in_prop: Área/proporción dentro de frontera agrícola
  - farming_out_ha / farming_out_prop: Área/proporción fuera de frontera agrícola
  - protected_ha / protected_prop: Área/proporción en zonas protegidas

Todas las entradas son GeoDataFrames y la salida es un DataFrame.

Public API:
  - spatial_metrics(plots, farming_areas=None, protected_areas=None, crs=None) -> pd.DataFrame

"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _area_ha(geom: BaseGeometry) -> float:
    """Calcula área de una geometría en hectáreas (asume CRS en metros)."""
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area / 10_000.0)


def _safe_prop(numerator: float, denominator: float) -> float:
    """División segura, retorna 0.0 si denominador <= 0."""
    if denominator is None or denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _intersect_area_ha(farm_geom: BaseGeometry, mask_gdf: gpd.GeoDataFrame) -> float:
    """
    Calcula área de intersección entre una geometría y un GeoDataFrame de referencia.
    Usa spatial index (sindex) para filtrar candidatos eficientemente.

    Args:
        farm_geom: Geometría del predio (en CRS proyectado, metros).
        mask_gdf: GeoDataFrame con polígonos de referencia (frontera/PNN).

    Returns:
        Área de intersección en hectáreas.
    """
    if mask_gdf is None or mask_gdf.empty:
        return 0.0

    if farm_geom is None or farm_geom.is_empty:
        return 0.0

    # Filtrar candidatos con spatial index
    try:
        sidx = mask_gdf.sindex
    except Exception:
        sidx = None

    if sidx is not None:
        cand_idx = list(sidx.intersection(farm_geom.bounds))
    else:
        cand_idx = list(range(len(mask_gdf)))

    if not cand_idx:
        return 0.0

    mask_sub = mask_gdf.iloc[cand_idx]
    mask_sub = mask_sub[mask_sub.intersects(farm_geom)]

    if mask_sub.empty:
        return 0.0

    # Calcular intersección acumulada (unión de todas las intersecciones
    # para evitar doble-conteo si los polígonos de referencia se solapan)
    covered = None
    for mg in mask_sub.geometry:
        try:
            inter = farm_geom.intersection(mg)
        except Exception:
            try:
                inter = farm_geom.buffer(0).intersection(mg.buffer(0))
            except Exception:
                continue

        if inter.is_empty:
            continue

        covered = inter if covered is None else covered.union(inter)

    if covered is None or covered.is_empty:
        return 0.0

    return _area_ha(covered)


def _ensure_projected_crs(
    gdf: gpd.GeoDataFrame,
    target_crs: str,
    label: str
) -> gpd.GeoDataFrame:
    """
    Asegura que un GeoDataFrame esté en el CRS objetivo (proyectado, metros).

    - Si no tiene CRS, asume que ya está en el target.
    - Si tiene otro CRS, lo reproyecta.

    Args:
        gdf: GeoDataFrame a verificar/reproyectar.
        target_crs: CRS destino (ej: "EPSG:3116").
        label: Etiqueta para mensajes de log.

    Returns:
        GeoDataFrame en el CRS destino.
    """
    if gdf.crs is None:
        logging.warning(f"{label}: sin CRS definido, asumiendo {target_crs}")
        gdf = gdf.set_crs(target_crs)
    elif str(gdf.crs).upper() != target_crs.upper():
        logging.info(f"{label}: reproyectando de {gdf.crs} a {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spatial_metrics(
    plots: gpd.GeoDataFrame,
    farming_areas: Optional[gpd.GeoDataFrame] = None,
    protected_areas: Optional[gpd.GeoDataFrame] = None,
    crs: str = "EPSG:3116",
    id_column: str = "id",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Calcula métricas espaciales para cada predio.

    Se calcula una sola vez y el resultado se reutiliza para todas las capas
    de deforestación, ya que frontera agrícola y áreas protegidas no cambian.

    Parameters
    ----------
    plots : geopandas.GeoDataFrame
        Predios/fincas con al menos una columna de ID y geometría (polígonos).
        CRS puede ser cualquiera; se reproyecta internamente al CRS indicado.
    farming_areas : geopandas.GeoDataFrame, optional
        Polígonos de frontera agrícola. Si None, se omiten esas métricas (0.0).
    protected_areas : geopandas.GeoDataFrame, optional
        Polígonos de áreas protegidas (PNN). Si None, se omiten esas métricas (0.0).
    crs : str, default "EPSG:3116"
        CRS proyectado en metros para cálculos de área. Todas las capas se
        reproyectan a este CRS.
    id_column : str, default "id"
        Nombre de la columna que identifica cada predio.
    show_progress : bool, default True
        Si True, muestra barra de progreso con tqdm.

    Returns
    -------
    pandas.DataFrame
        Una fila por predio con:
          - id: identificador del predio
          - total_ha: área total en hectáreas
          - farming_in_ha: área dentro de frontera agrícola (ha)
          - farming_in_prop: proporción en frontera [0, 1]
          - farming_out_ha: área fuera de frontera agrícola (ha)
          - farming_out_prop: proporción fuera de frontera [0, 1]
          - protected_ha: área en zonas protegidas (ha)
          - protected_prop: proporción en zonas protegidas [0, 1]

    Raises
    ------
    ValueError
        Si `plots` no contiene la columna id_column o geometry.

    Examples
    --------
    >>> import geopandas as gpd
    >>> plots = gpd.read_file("farms.geojson")
    >>> farming = gpd.read_file("frontera_agricola.shp")
    >>> protected = gpd.read_file("pnn_areas.shp")
    >>> metrics = spatial_metrics(plots, farming_areas=farming, protected_areas=protected)
    >>> print(metrics.head())
    """
    # -------------------------------------------------------------------------
    # Validaciones
    # -------------------------------------------------------------------------
    if id_column not in plots.columns:
        raise ValueError(
            f"plots debe contener la columna '{id_column}'. "
            f"Columnas disponibles: {list(plots.columns)}"
        )

    if "geometry" not in plots.columns and plots.geometry.name not in plots.columns:
        raise ValueError("plots debe contener una columna de geometría.")

    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Preparar datos: reproyectar al CRS objetivo
    # -------------------------------------------------------------------------
    plots_proj = _ensure_projected_crs(
        plots[[id_column, "geometry"]].copy(),
        target_crs=crs,
        label="plots"
    )

    # Eliminar filas sin geometría válida
    plots_proj = plots_proj[plots_proj.geometry.notnull()].reset_index(drop=True)

    if len(plots_proj) == 0:
        logging.warning("No hay predios con geometría válida.")
        return pd.DataFrame(columns=[
            "id", "total_ha",
            "farming_in_ha", "farming_in_prop",
            "farming_out_ha", "farming_out_prop",
            "protected_ha", "protected_prop",
        ])

    # Reproyectar capas de referencia
    farming_proj = None
    if farming_areas is not None and not farming_areas.empty:
        farming_proj = _ensure_projected_crs(farming_areas.copy(), crs, "farming_areas")

    protected_proj = None
    if protected_areas is not None and not protected_areas.empty:
        protected_proj = _ensure_projected_crs(protected_areas.copy(), crs, "protected_areas")

    if farming_proj is None and protected_proj is None:
        logging.warning(
            "No se proporcionaron capas de referencia (farming_areas ni protected_areas). "
            "Solo se calculará total_ha."
        )

    # -------------------------------------------------------------------------
    # Calcular métricas por predio
    # -------------------------------------------------------------------------
    n_plots = len(plots_proj)
    print(f"📐 Calculando métricas espaciales para {n_plots:,} predios...")

    rows = []

    iterator = plots_proj.iterrows()
    if show_progress:
        iterator = tqdm(
            plots_proj.iterrows(),
            total=n_plots,
            desc="Métricas espaciales",
            unit="predio",
        )

    for _, row in iterator:
        plot_id = row[id_column]
        geom = row.geometry

        # Área total
        total_ha = _area_ha(geom)

        # Frontera agrícola
        farming_in_ha = _intersect_area_ha(geom, farming_proj)
        farming_out_ha = max(total_ha - farming_in_ha, 0.0)

        # Áreas protegidas
        protected_ha = _intersect_area_ha(geom, protected_proj)

        rows.append({
            "id": plot_id,
            "total_ha": round(total_ha, 4),
            "farming_in_ha": round(farming_in_ha, 4),
            "farming_in_prop": round(_safe_prop(farming_in_ha, total_ha), 6),
            "farming_out_ha": round(farming_out_ha, 4),
            "farming_out_prop": round(_safe_prop(farming_out_ha, total_ha), 6),
            "protected_ha": round(protected_ha, 4),
            "protected_prop": round(_safe_prop(protected_ha, total_ha), 6),
        })

    elapsed = time.perf_counter() - t0
    print(f"✅ Métricas calculadas: {n_plots:,} predios en {elapsed:.2f}s")

    # -------------------------------------------------------------------------
    # Construir DataFrame resultado
    # -------------------------------------------------------------------------
    result = pd.DataFrame(rows)

    # Clamp proporciones a [0, 1]
    for col in ["farming_in_prop", "farming_out_prop", "protected_prop"]:
        result[col] = result[col].clip(lower=0.0, upper=1.0)

    return result
