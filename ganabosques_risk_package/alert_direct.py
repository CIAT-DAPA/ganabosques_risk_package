# -*- coding: utf-8 -*-
"""
Cálculo de alertas directas de deforestación para predios (plots/farms).

Intersecta cada predio con un raster de deforestación para determinar:
  - deforested_ha: Hectáreas deforestadas dentro del predio
  - deforested_prop: Proporción del predio deforestada [0, 1]
  - direct_alert: Booleano indicando si hay deforestación

Dos modos de cálculo:
  - Rápido (default): Cuenta píxeles completos cuyo centro cae en el predio.
  - Preciso (use_precise_area=True): Vectoriza los píxeles deforestados y calcula
    la intersección geométrica real con el predio. Más lento pero exacto en bordes.

Opcionalmente recibe el DataFrame de métricas espaciales (de spatial_metrics())
para agregarlo al resultado, evitando recalcular para cada capa de deforestación.

Public API:
  - alert_direct(plots, deforestation_raster, ...) -> pd.DataFrame

"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rasterio_mask
from rasterio.features import shapes as rasterio_shapes
from rasterio.vrt import WarpedVRT
from shapely.geometry import mapping, shape
from shapely.ops import unary_union, transform as shapely_transform
from pyproj import Transformer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _crs_eq(a: str, b: str) -> bool:
    """Compara dos cadenas CRS de forma flexible."""
    return a.strip().upper().replace(" ", "") == b.strip().upper().replace(" ", "")


def _area_ha(geom) -> float:
    """Área de una geometría en hectáreas (asume CRS en metros)."""
    if geom is None or geom.is_empty:
        return 0.0
    return float(geom.area) / 10_000.0


def _pixel_area_m2_approx_4326(src) -> float:
    """
    Aproxima área de un píxel en m² cuando el raster está en EPSG:4326 (grados).
    Usa la latitud media del raster para la conversión.
    """
    try:
        t = src.transform
        pixel_deg_x = abs(t.a)
        pixel_deg_y = abs(t.e)
        left, bottom, right, top = src.bounds
        center_lat = (top + bottom) / 2.0
        lat_rad = math.radians(center_lat)
        m_per_deg_lat = 110574.0
        m_per_deg_lon = 111320.0 * math.cos(lat_rad)
        return abs(pixel_deg_x * pixel_deg_y * m_per_deg_lon * m_per_deg_lat)
    except Exception:
        return abs(pixel_deg_x * pixel_deg_y * (111000.0 ** 2))


def _open_raster(path: str, target_crs: str = "EPSG:3116"):
    """
    Abre un raster. Si su CRS no coincide con target_crs, devuelve un
    WarpedVRT que lo reproyecta virtualmente al target_crs.
    """
    src = rasterio.open(path)
    src_crs = src.crs
    if src_crs is None:
        raise ValueError(f"Raster {path} no tiene CRS definido.")
    if _crs_eq(src_crs.to_string(), target_crs):
        return src
    else:
        return WarpedVRT(src, crs=target_crs)


def _calculate_deforestation_for_plot(
    src,
    geom,
    deforest_value: int,
    crs: str,
    use_precise_area: bool = False,
) -> Tuple[bool, float, float]:
    """
    Calcula métricas de deforestación para un predio individual.

    Args:
        src: Dataset de rasterio abierto (o WarpedVRT).
        geom: Geometría del predio en CRS proyectado (metros).
        deforest_value: Valor del píxel que indica deforestación.
        crs: CRS proyectado del predio (ej: "EPSG:3116").
        use_precise_area: Si True, usa intersección geométrica exacta.

    Returns:
        (intersectó, defo_ha, proporción_0_1)
    """
    try:
        src_crs = src.crs.to_string() if src.crs else None
        geom_for_mask = geom

        # Si el raster tiene CRS distinto, transformar geometría al CRS del raster
        if src_crs and not _crs_eq(src_crs, crs):
            transformer = Transformer.from_crs(crs, src_crs, always_xy=True)
            func = lambda x, y, z=None: transformer.transform(x, y)
            geom_for_mask = shapely_transform(func, geom)

        # Extraer píxeles que tocan el predio
        # all_touched=True con precise-area: incluir TODOS los píxeles que tocan
        # el polígono, el cálculo geométrico se encargará de las fracciones
        out_image, out_transform = rasterio_mask(
            src, [mapping(geom_for_mask)],
            crop=True, filled=False,
            all_touched=use_precise_area
        )
        arr = out_image[0]  # MaskedArray
        valid = ~np.ma.getmaskarray(arr)
        m = valid & (arr.data == deforest_value)
        cnt = int(np.count_nonzero(m))
        if cnt == 0:
            return False, 0.0, 0.0

        # Calcular área deforestada
        if use_precise_area:
            # 🎯 MÉTODO PRECISO: Vectorizar píxeles deforestados e intersectar
            # geométricamente con el predio para obtener área exacta
            pixel_polygons = []
            for geom_json, value in rasterio_shapes(
                arr.data, mask=m, transform=out_transform
            ):
                if value == deforest_value:
                    pixel_polygons.append(shape(geom_json))

            if not pixel_polygons:
                return False, 0.0, 0.0

            # Unir todos los píxeles deforestados en una geometría
            union_pixels = unary_union(pixel_polygons)

            # Intersección geométrica real con el predio
            # (usa geom_for_mask que está en el CRS del raster)
            intersection = geom_for_mask.intersection(union_pixels)

            if intersection.is_empty:
                return False, 0.0, 0.0

            # Si el raster está en 4326, necesitamos reproyectar para obtener área en m²
            if src_crs and _crs_eq(src_crs, "EPSG:4326"):
                # Reproyectar intersección a CRS proyectado para medir área
                transformer_back = Transformer.from_crs(src_crs, crs, always_xy=True)
                func_back = lambda x, y, z=None: transformer_back.transform(x, y)
                intersection_proj = shapely_transform(func_back, intersection)
                defo_ha = float(intersection_proj.area) / 10_000.0
            else:
                defo_ha = float(intersection.area) / 10_000.0
        else:
            # ⚡ MÉTODO RÁPIDO: Contar píxeles completos
            if src_crs and _crs_eq(src_crs, "EPSG:4326"):
                pixel_area = _pixel_area_m2_approx_4326(src)
            else:
                pixel_area = abs(src.transform.a * src.transform.e)
            defo_ha = cnt * pixel_area / 10_000.0

        # Proporción respecto al área total del predio
        geom_ha = _area_ha(geom)
        if geom_ha <= 0:
            prop = 0.0
        else:
            prop = defo_ha / geom_ha

        # Limitar a [0, 1]
        prop = max(0.0, min(float(prop), 1.0))

        return True, float(defo_ha), prop

    except Exception as e:
        logging.warning(f"Error calculando deforestación: {e}")
        return False, 0.0, 0.0


def _ensure_projected_crs(
    gdf: gpd.GeoDataFrame, target_crs: str, label: str
) -> gpd.GeoDataFrame:
    """Asegura que un GeoDataFrame esté en el CRS objetivo."""
    if gdf.crs is None:
        logging.warning(f"{label}: sin CRS, asumiendo {target_crs}")
        gdf = gdf.set_crs(target_crs)
    elif not _crs_eq(str(gdf.crs), target_crs):
        logging.info(f"{label}: reproyectando de {gdf.crs} a {target_crs}")
        gdf = gdf.to_crs(target_crs)
    return gdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def alert_direct(
    plots: gpd.GeoDataFrame,
    deforestation_raster: str,
    deforestation_value: int = 2,
    metrics_df: Optional[pd.DataFrame] = None,
    crs: str = "EPSG:3116",
    id_column: str = "id",
    use_precise_area: bool = False,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Calcula alertas directas de deforestación para cada predio.

    Intersecta cada predio (polígono) con un raster de deforestación para
    determinar hectáreas y proporción deforestada.

    Parameters
    ----------
    plots : geopandas.GeoDataFrame
        Predios con al menos columna de ID y geometría (polígonos).
        CRS puede ser cualquiera; se reproyecta al CRS indicado.
    deforestation_raster : str
        Ruta al archivo raster de deforestación (GeoTIFF, etc.).
    deforestation_value : int, default 2
        Valor de píxel que representa deforestación en el raster.
    metrics_df : pandas.DataFrame, optional
        DataFrame resultado de ``spatial_metrics()`` con columnas:
        id, total_ha, farming_in_ha, farming_in_prop, farming_out_ha,
        farming_out_prop, protected_ha, protected_prop.
        Si se proporciona, se hace merge con el resultado por la columna ``id``.
    crs : str, default "EPSG:3116"
        CRS proyectado en metros para cálculos de área. Los predios se
        reproyectan a este CRS internamente.
    id_column : str, default "id"
        Nombre de la columna que identifica cada predio.
    use_precise_area : bool, default False
        Si True, vectoriza los píxeles deforestados y calcula la intersección
        geométrica real con el predio (más lento pero preciso en bordes).
        Si False, cuenta píxeles completos cuyo centro cae en el predio.
    show_progress : bool, default True
        Mostrar barra de progreso con tqdm.

    Returns
    -------
    pandas.DataFrame
        Una fila por predio con:
          - id: identificador del predio
          - deforested_ha: hectáreas deforestadas (float)
          - deforested_prop: proporción deforestada [0, 1] (float)
          - direct_alert: True si hay deforestación (bool)
          - [columnas de metrics_df si se proporcionó]

    Raises
    ------
    ValueError
        Si ``plots`` no contiene id_column o geometría.
    FileNotFoundError
        Si ``deforestation_raster`` no existe.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from ganabosques_risk_package.spatial_metrics import spatial_metrics
    >>> from ganabosques_risk_package.alert_direct import alert_direct
    >>>
    >>> plots = gpd.read_file("farms.geojson")
    >>> # Calcular métricas una sola vez
    >>> metrics = spatial_metrics(plots, farming_areas=farming, protected_areas=protected)
    >>> # Calcular alertas para cada capa de deforestación
    >>> for raster_path in raster_list:
    ...     alerts = alert_direct(plots, raster_path, metrics_df=metrics)
    ...     alerts.to_csv(f"alerts_{raster_path.stem}.csv", index=False)
    """
    # -------------------------------------------------------------------------
    # Validaciones
    # -------------------------------------------------------------------------
    if id_column not in plots.columns:
        raise ValueError(
            f"plots debe contener la columna '{id_column}'. "
            f"Columnas disponibles: {list(plots.columns)}"
        )

    import os
    if not os.path.isfile(deforestation_raster):
        raise FileNotFoundError(
            f"Raster de deforestación no encontrado: {deforestation_raster}"
        )

    t0 = time.perf_counter()

    # -------------------------------------------------------------------------
    # Preparar datos
    # -------------------------------------------------------------------------
    plots_proj = _ensure_projected_crs(
        plots[[id_column, "geometry"]].copy(), crs, "plots"
    )
    plots_proj = plots_proj[plots_proj.geometry.notnull()].reset_index(drop=True)

    n_plots = len(plots_proj)
    if n_plots == 0:
        logging.warning("No hay predios con geometría válida.")
        return _empty_result(metrics_df is not None)

    # -------------------------------------------------------------------------
    # Abrir raster
    # -------------------------------------------------------------------------
    raster_src = _open_raster(deforestation_raster, target_crs=crs)

    mode_label = "preciso (intersección geométrica)" if use_precise_area else "rápido (píxeles completos)"
    print(f"🔍 Calculando alertas directas para {n_plots:,} predios [{mode_label}]")

    # -------------------------------------------------------------------------
    # Procesar cada predio
    # -------------------------------------------------------------------------
    results = []

    iterator = plots_proj.iterrows()
    if show_progress:
        iterator = tqdm(
            plots_proj.iterrows(),
            total=n_plots,
            desc="Alertas directas",
            unit="predio",
        )

    for _, row in iterator:
        plot_id = row[id_column]
        geom = row.geometry

        intersected, defo_ha, defo_prop = _calculate_deforestation_for_plot(
            src=raster_src,
            geom=geom,
            deforest_value=deforestation_value,
            crs=crs,
            use_precise_area=use_precise_area,
        )

        results.append({
            "id": plot_id,
            "deforested_ha": round(defo_ha, 4),
            "deforested_prop": round(defo_prop, 6),
            "direct_alert": bool(intersected),
        })

    # Cerrar raster
    try:
        raster_src.close()
    except Exception:
        pass

    elapsed = time.perf_counter() - t0
    print(f"✅ Alertas directas: {n_plots:,} predios en {elapsed:.2f}s")

    # -------------------------------------------------------------------------
    # Construir DataFrame resultado
    # -------------------------------------------------------------------------
    result_df = pd.DataFrame(results)

    # Clamp proporción
    result_df["deforested_prop"] = result_df["deforested_prop"].clip(
        lower=0.0, upper=1.0
    )

    # -------------------------------------------------------------------------
    # Merge con métricas espaciales (si se proporcionaron)
    # -------------------------------------------------------------------------
    if metrics_df is not None and not metrics_df.empty:
        # Asegurar que la columna id sea del mismo tipo
        result_df["id"] = result_df["id"].astype(str)
        metrics_merge = metrics_df.copy()
        metrics_merge["id"] = metrics_merge["id"].astype(str)

        result_df = result_df.merge(metrics_merge, on="id", how="left")

    # Estadísticas rápidas
    n_alert = result_df["direct_alert"].sum()
    print(f"   📊 Predios con alerta: {n_alert:,} / {n_plots:,} ({100*n_alert/n_plots:.1f}%)")
    if n_alert > 0:
        avg_ha = result_df.loc[result_df["direct_alert"], "deforested_ha"].mean()
        print(f"   📊 Deforestación promedio (con alerta): {avg_ha:.4f} ha")

    return result_df


def _empty_result(with_metrics: bool = False) -> pd.DataFrame:
    """Retorna DataFrame vacío con las columnas esperadas."""
    cols = ["id", "deforested_ha", "deforested_prop", "direct_alert"]
    if with_metrics:
        cols += [
            "total_ha", "farming_in_ha", "farming_in_prop",
            "farming_out_ha", "farming_out_prop",
            "protected_ha", "protected_prop",
        ]
    return pd.DataFrame(columns=cols)
