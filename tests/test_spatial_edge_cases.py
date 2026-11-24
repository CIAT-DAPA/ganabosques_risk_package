"""Casos límite de alert_direct y spatial_metrics con un raster sintético.

El raster se construye en cada corrida (EPSG:3116, píxeles de 30 m, 20x20) para
que los valores esperados se puedan calcular a mano:

  - valor 2 (deforestación) en un bloque interior de 4x4 píxeles
    (columnas 2-5, filas 2-5) -> 16 px = 1.44 ha
  - valor 2 en la columna 10, filas 2-5 (4 px), que el predio de prueba solo
    cubre en sus primeros 10 m (1/3 de cada píxel)
  - el resto del raster vale 1

Predio de prueba ``PLOT``: columnas 1-9 completas + 10 m de la columna 10,
filas 1-7 -> 280 m x 210 m = 5.88 ha.
"""

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box

from ganabosques_risk_package import alert_direct, spatial_metrics

X0, Y0, PX = 1_000_000.0, 1_000_000.0, 30.0
PX_HA = PX * PX / 10_000  # 0.09 ha


def cell(col, row):
    """Polígono de un píxel del raster sintético."""
    return box(X0 + col * PX, Y0 - (row + 1) * PX, X0 + (col + 1) * PX, Y0 - row * PX)


XMIN, XMAX = X0 + 1 * PX, X0 + 10 * PX + 10
YMIN, YMAX = Y0 - 8 * PX, Y0 - 1 * PX
PLOT = box(XMIN, YMIN, XMAX, YMAX)
PLOT_HA = 5.88
INTERIOR_HA = 16 * PX_HA                # 1.44
EDGE_HA = 4 * (10 * PX) / 10_000        # 4 píxeles x 300 m² = 0.12


def plots(geoms, ids=None, crs="EPSG:3116"):
    ids = ids or [f"p{i}" for i in range(len(geoms))]
    return gpd.GeoDataFrame({"id": ids}, geometry=geoms, crs=crs)


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, show_progress=False, **kwargs)


def row(df):
    return df.iloc[0]


class SyntheticRasterMixin:

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="ganabosques_test_"))
        arr = np.ones((20, 20), dtype="uint8")
        arr[2:6, 2:6] = 2
        arr[2:6, 10] = 2
        cls.raster = str(cls.tmp / "defo_3116.tif")
        with rasterio.open(
            cls.raster, "w", driver="GTiff", height=20, width=20, count=1,
            dtype="uint8", crs="EPSG:3116", transform=from_origin(X0, Y0, PX, PX), nodata=0,
        ) as dst:
            dst.write(arr, 1)

        # Mismo dato reproyectado a EPSG:4326 (vecino más cercano)
        cls.raster_4326 = str(cls.tmp / "defo_4326.tif")
        with rasterio.open(cls.raster) as src:
            t, w, h = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds)
            profile = src.profile | {"crs": "EPSG:4326", "transform": t, "width": w, "height": h}
            with rasterio.open(cls.raster_4326, "w", **profile) as dst:
                reproject(rasterio.band(src, 1), rasterio.band(dst, 1), resampling=Resampling.nearest)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestAlertDirectEdgeCases(SyntheticRasterMixin, unittest.TestCase):

    # ------------------------------------------------------------ modos de área

    def test_fast_mode_counts_only_pixels_with_center_inside(self):
        r = row(quiet(alert_direct, plots([PLOT]), self.raster))
        # Los píxeles de la columna 10 tienen el centro fuera del predio
        self.assertAlmostEqual(r["deforested_ha"], INTERIOR_HA, places=4)
        self.assertAlmostEqual(r["deforested_prop"], INTERIOR_HA / PLOT_HA, places=5)
        self.assertTrue(r["direct_alert"])

    def test_precise_mode_adds_partial_border_pixels(self):
        r = row(quiet(alert_direct, plots([PLOT]), self.raster, use_precise_area=True))
        self.assertAlmostEqual(r["deforested_ha"], INTERIOR_HA + EDGE_HA, places=4)
        self.assertAlmostEqual(r["deforested_prop"], (INTERIOR_HA + EDGE_HA) / PLOT_HA, places=5)

    def test_modes_agree_when_pixels_are_fully_inside(self):
        inner = box(X0 + 2 * PX, Y0 - 6 * PX, X0 + 6 * PX, Y0 - 2 * PX)  # exactamente el bloque 4x4
        fast = row(quiet(alert_direct, plots([inner]), self.raster))
        precise = row(quiet(alert_direct, plots([inner]), self.raster, use_precise_area=True))
        self.assertAlmostEqual(fast["deforested_ha"], INTERIOR_HA, places=4)
        self.assertAlmostEqual(precise["deforested_ha"], INTERIOR_HA, places=4)
        self.assertAlmostEqual(fast["deforested_prop"], 1.0, places=5)

    def test_deforestation_value_parameter(self):
        # 9 x 7 = 63 centros dentro del predio, 16 son deforestación -> 47 con valor 1
        r = row(quiet(alert_direct, plots([PLOT]), self.raster, deforestation_value=1))
        self.assertAlmostEqual(r["deforested_ha"], 47 * PX_HA, places=4)

    def test_plot_without_deforestation(self):
        clean = box(X0 + 12 * PX, Y0 - 18 * PX, X0 + 18 * PX, Y0 - 12 * PX)
        r = row(quiet(alert_direct, plots([clean]), self.raster))
        self.assertFalse(r["direct_alert"])
        self.assertEqual(r["deforested_ha"], 0.0)

    # ------------------------------------------------------------------ CRS

    def test_raster_in_other_crs(self):
        for precise in (False, True):
            with self.subTest(use_precise_area=precise):
                ref = row(quiet(alert_direct, plots([PLOT]), self.raster, use_precise_area=precise))
                r = row(quiet(alert_direct, plots([PLOT]), self.raster_4326, use_precise_area=precise))
                self.assertTrue(r["direct_alert"])
                # Doble remuestreo (3116 -> 4326 -> 3116): tolerancia del 5 %
                self.assertAlmostEqual(r["deforested_ha"], ref["deforested_ha"], delta=0.05 * ref["deforested_ha"])

    def test_plots_in_other_crs_are_reprojected(self):
        r = row(quiet(alert_direct, plots([PLOT]).to_crs("EPSG:4326"), self.raster))
        self.assertAlmostEqual(r["deforested_ha"], INTERIOR_HA, places=4)
        self.assertAlmostEqual(r["deforested_prop"], INTERIOR_HA / PLOT_HA, places=4)

    def test_plots_without_crs_assume_target_crs(self):
        gdf = gpd.GeoDataFrame({"id": ["a"]}, geometry=[PLOT])
        with self.assertLogs(level="WARNING"):
            r = row(quiet(alert_direct, gdf, self.raster))
        self.assertAlmostEqual(r["deforested_ha"], INTERIOR_HA, places=4)

    # ------------------------------------------------------------ geometrías

    def test_multipolygon(self):
        # Un píxel del bloque interior + un píxel de la columna 10, ambos completos
        multi = MultiPolygon([cell(2, 2), cell(10, 3)])
        r = row(quiet(alert_direct, plots([multi]), self.raster))
        self.assertAlmostEqual(r["deforested_ha"], 2 * PX_HA, places=4)
        self.assertAlmostEqual(r["deforested_prop"], 1.0, places=5)

    def test_geometry_collection_uses_only_polygon_parts(self):
        gc = GeometryCollection([cell(2, 2), LineString([(X0, Y0), (X0 + 1, Y0 - 1)])])
        r = row(quiet(alert_direct, plots([gc]), self.raster))
        self.assertAlmostEqual(r["deforested_ha"], PX_HA, places=4)
        self.assertTrue(r["direct_alert"])

    def test_geometries_without_area_give_no_alert(self):
        for name, geom in [
            ("empty", Polygon()),
            ("line", LineString([(XMIN, YMIN), (XMAX, YMAX)])),
        ]:
            with self.subTest(name), self.assertLogs(level="WARNING"):
                r = row(quiet(alert_direct, plots([geom], [name]), self.raster))
                self.assertFalse(r["direct_alert"])
                self.assertEqual(r["deforested_ha"], 0.0)

    def test_plot_outside_raster_gives_no_alert(self):
        far = box(X0 + 10_000, Y0 + 10_000, X0 + 10_100, Y0 + 10_100)
        with self.assertLogs(level="WARNING"):
            r = row(quiet(alert_direct, plots([far]), self.raster))
        self.assertFalse(r["direct_alert"])

    def test_overlapping_plots_are_evaluated_independently(self):
        # Fincas distintas que se solapan (problema de datos): cada una recibe
        # su propia deforestación. Al SUMAR entre fincas, el área compartida
        # cuenta una vez por cada finca.
        inner = box(X0 + 2 * PX, Y0 - 6 * PX, X0 + 6 * PX, Y0 - 2 * PX)  # dentro de PLOT
        out = quiet(alert_direct, plots([PLOT, inner], ["GRANDE", "CHICA"]), self.raster).set_index("id")
        self.assertAlmostEqual(out.loc["GRANDE", "deforested_ha"], INTERIOR_HA, places=4)
        self.assertAlmostEqual(out.loc["CHICA", "deforested_ha"], INTERIOR_HA, places=4)
        self.assertTrue(out["direct_alert"].all())

    def test_several_plots_keep_their_ids(self):
        out = quiet(alert_direct, plots([PLOT, cell(10, 3), cell(15, 15)], ["A", "B", "C"]), self.raster)
        self.assertEqual(out["id"].tolist(), ["A", "B", "C"])
        self.assertEqual(out["direct_alert"].tolist(), [True, True, False])


    # ------------------------------------------------------------ resultados vacíos

    def test_no_valid_geometries_returns_empty_frame(self):
        with self.assertLogs(level="WARNING"):
            out = quiet(alert_direct, plots([None]), self.raster)
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), ["id", "deforested_ha", "deforested_prop", "direct_alert"])

    def test_no_valid_geometries_with_metrics_keeps_metric_columns(self):
        metrics = quiet(spatial_metrics, plots([PLOT]))
        with self.assertLogs(level="WARNING"):
            out = quiet(alert_direct, plots([None]), self.raster, metrics_df=metrics)
        self.assertIn("total_ha", out.columns)


class TestSpatialMetricsEdgeCases(unittest.TestCase):

    # Capas de referencia dentro del predio
    REF1 = box(X0 + 1 * PX, Y0 - 8 * PX, X0 + 5 * PX, Y0 - 1 * PX)   # 120 m x 210 m = 2.52 ha
    REF2 = box(X0 + 3 * PX, Y0 - 8 * PX, X0 + 7 * PX, Y0 - 1 * PX)   # solapa 60 m con REF1
    UNION_HA = 3.78                                                  # 180 m x 210 m

    def metrics(self, plot_geoms, farming=None, protected=None, **kwargs):
        def layer(geoms):
            return None if geoms is None else plots(geoms)
        return row(quiet(spatial_metrics, plots(plot_geoms), layer(farming), layer(protected), **kwargs))

    def test_areas_and_proportions(self):
        r = self.metrics([PLOT], farming=[self.REF1], protected=[self.REF1])
        self.assertAlmostEqual(r["total_ha"], PLOT_HA, places=4)
        self.assertAlmostEqual(r["farming_in_ha"], 2.52, places=4)
        self.assertAlmostEqual(r["farming_out_ha"], PLOT_HA - 2.52, places=4)
        self.assertAlmostEqual(r["protected_prop"], 2.52 / PLOT_HA, places=5)

    def test_overlapping_reference_polygons_are_not_double_counted(self):
        r = self.metrics([PLOT], farming=[self.REF1, self.REF2], protected=[self.REF1, self.REF1])
        self.assertAlmostEqual(r["farming_in_ha"], self.UNION_HA, places=4)
        self.assertAlmostEqual(r["protected_ha"], 2.52, places=4)

    def test_reference_layer_far_away(self):
        r = self.metrics([PLOT], farming=[box(0, 0, 10, 10)])
        self.assertEqual(r["farming_in_ha"], 0.0)
        self.assertEqual(r["farming_out_prop"], 1.0)

    def test_reference_larger_than_plot_caps_at_plot_area(self):
        r = self.metrics([PLOT], protected=[PLOT.buffer(500)])
        self.assertAlmostEqual(r["protected_ha"], PLOT_HA, places=4)
        self.assertAlmostEqual(r["protected_prop"], 1.0, places=5)

    def test_multipolygon_plot(self):
        r = self.metrics([MultiPolygon([cell(2, 2), cell(10, 3)])], farming=[cell(2, 2)])
        self.assertAlmostEqual(r["total_ha"], 2 * PX_HA, places=4)
        self.assertAlmostEqual(r["farming_in_prop"], 0.5, places=5)

    def test_layers_in_other_crs(self):
        plot_4326 = plots([PLOT]).to_crs("EPSG:4326")
        ref_4326 = plots([self.REF1]).to_crs("EPSG:4326")
        r = row(quiet(spatial_metrics, plot_4326, ref_4326, None))
        self.assertAlmostEqual(r["total_ha"], PLOT_HA, places=2)
        self.assertAlmostEqual(r["farming_in_ha"], 2.52, places=2)

    def test_reference_layer_without_crs(self):
        ref = gpd.GeoDataFrame(geometry=[self.REF1])
        with self.assertLogs(level="WARNING"):
            r = row(quiet(spatial_metrics, plots([PLOT]), ref, None))
        self.assertAlmostEqual(r["farming_in_ha"], 2.52, places=4)

    def test_other_projected_crs_is_accepted(self):
        # Cualquier CRS proyectado en metros sirve (aquí UTM 18N)
        r = row(quiet(spatial_metrics, plots([PLOT]), crs="EPSG:32618"))
        self.assertAlmostEqual(r["total_ha"], PLOT_HA, delta=0.01 * PLOT_HA)

    def test_overlapping_plots_are_evaluated_independently(self):
        inner = box(X0 + 2 * PX, Y0 - 6 * PX, X0 + 6 * PX, Y0 - 2 * PX)
        out = quiet(spatial_metrics, plots([PLOT, inner], ["GRANDE", "CHICA"]), None, plots([PLOT])).set_index("id")
        self.assertAlmostEqual(out.loc["GRANDE", "protected_ha"], PLOT_HA, places=4)
        self.assertAlmostEqual(out.loc["CHICA", "protected_ha"], 16 * PX_HA, places=4)

    def test_no_valid_geometries_returns_empty_frame(self):
        with self.assertLogs(level="WARNING"):
            out = quiet(spatial_metrics, plots([None]))
        self.assertTrue(out.empty)
        self.assertIn("total_ha", out.columns)


if __name__ == "__main__":
    unittest.main()
