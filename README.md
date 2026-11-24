# Ganabosques Risk Package

The **Ganabosques Risk Package** computes **deforestation risk for cattle farms** (plots) and propagates it to the companies connected to them. Every function receives and returns pandas DataFrames.

Pipeline:

```
spatial_metrics ─┐
                 ├─► alert_direct ─► alert_indirect ─► total_risk ─┬─► alert_enterprise   (companies, via cattle movements)
deforestation ───┘                                                 ├─► supplier_risk       (companies, via supplier relations)
raster                                                             └─► calculate_alert     (any entity: ADM3, associations, …)
```

---

## 📦 Installation

```bash
pip install git+https://github.com/CIAT-DAPA/ganabosques_risk_package.git
```

Upgrade / uninstall:

```bash
pip install --upgrade git+https://github.com/CIAT-DAPA/ganabosques_risk_package.git
pip uninstall ganabosques_risk_package
```

Requires Python ≥ 3.10. Runtime dependencies: geopandas, rasterio, shapely, pyproj, numpy, pandas (< 3), tqdm.

---

## 🚀 Quick start

```python
import geopandas as gpd
import pandas as pd
from ganabosques_risk_package import (
    spatial_metrics, alert_direct, alert_indirect, total_risk,
    alert_enterprise, supplier_risk, supplier_risk_summary,
)

plots = gpd.read_file("data/plots.geojson")                  # columns: id, geometry
farming = gpd.read_file("data/frontera_agricola.shp")
protected = gpd.read_file("data/pnn.shp")
movements = pd.read_csv("data/movements.csv")                # origen_id, destination_id, tipo_origen, tipo_destino
suppliers = pd.read_csv("data/suppliers.csv")                # enterprise_id, farm_id, years

# 1. Spatial metrics: computed once, reused for every deforestation layer
metrics = spatial_metrics(plots, farming_areas=farming, protected_areas=protected)

# 2. Direct alerts for one deforestation raster
direct = alert_direct(plots, "data/deforestation_2023.tif", metrics_df=metrics)

# 3. Indirect alerts through cattle movements
indirect = alert_indirect(direct, movements)

# 4. Consolidated risk per farm
risk = total_risk(direct, indirect)

# 5. Company risk
ent_alerts = alert_enterprise(risk, movements)
sup_risk = supplier_risk(risk, suppliers)
print(supplier_risk_summary(sup_risk))
```

---

## 🧠 Public functions

All functions accept `show_progress=False` to silence the console summary.

**Overlapping plots:** each plot is evaluated independently. If two plots with different IDs overlap, the shared area counts for both, so sums across plots (e.g. `total_deforested_ha`, `deforested_area_sum`) count it once per plot.

### `spatial_metrics(plots, farming_areas=None, protected_areas=None, crs="EPSG:3116", id_column="id")`

Area of each plot inside/outside the agricultural frontier and inside protected areas. All layers are reprojected to `crs` (a projected CRS in metres).

Output: `id, total_ha, farming_in_ha, farming_in_prop, farming_out_ha, farming_out_prop, protected_ha, protected_prop`.

### `alert_direct(plots, deforestation_raster, deforestation_value=2, metrics_df=None, crs="EPSG:3116", id_column="id", use_precise_area=False)`

Intersects each plot with a deforestation raster.

* Default mode counts pixels whose centre falls inside the plot.
* `use_precise_area=True` vectorises the deforested pixels and intersects them with the plot (slower, exact at the borders).
* If `metrics_df` is given, its columns are merged into the output.

Output: `id, deforested_ha, deforested_prop, direct_alert` (+ metric columns).

### `alert_indirect(alert_direct_df, movements_df, id_column="id", normalize_ids=True)`

One-hop propagation through movements (`origen_id → destination_id`):

| Column | Meaning |
|---|---|
| `n_in` / `n_out` | movements received / sent by the plot |
| `n_indirect_in` | movements **received from** a plot with direct alert |
| `n_indirect_out` | movements **sent to** a plot with direct alert |
| `indirect_alert_in` / `indirect_alert_out` | `n_indirect_in > 0` / `n_indirect_out > 0` |

With `normalize_ids=True` (default), IDs are normalised **for matching only**: `1`, `1.0`, `"1"` and `"0001"` are the same plot. This matters because pandas reads an integer ID column with missing values as float. The output `id` keeps the original value.

### `total_risk(direct_df, indirect_df=None, metrics_df=None, id_column="id")`

Joins direct alerts, indirect alerts and spatial metrics into one row per plot.

* Without `indirect_df`, the indirect columns are filled with `"no_info"`.
* If `direct_df` and `indirect_df` do not share the same IDs, a warning is logged and missing values are filled as **no alert** (`False` / `0`). Missing values are never treated as alerts.

### `alert_enterprise(total_risk_df, movements_df, id_column="id", enterprise_types=None, alert_columns=None, normalize_ids=True)`

Companies (slaughterhouses, cattle fairs, collection centres, …) that exchanged cattle with a farm with any alert.

* `typemove="in"`: alerted farm → company. `typemove="out"`: company → alerted farm.
* Upper-case SIT column names (`SIT_CODE_ORIGEN`, `TIPO_DESTINO`, `PRODUCER_ID_DESTINO`, …) are renamed automatically.
* If `tipo_origen`/`tipo_destino` are missing, any ID that is not in `total_risk_df` is considered a company (a warning is logged).

Output: `idpro, id_farm, typemove, enterprise_type, farm_has_direct_alert, farm_has_indirect_alert`.

### `supplier_risk(total_risk_df, suppliers_df, period=None, period_type=None, filter_by_year=False, ...)` and `supplier_risk_summary(df)`

Company risk from formal supplier relations (company ↔ farm), for companies without movement data.

* `supplier_risk` returns one row per relation whose farm has risk (`has_risk, risk_direct, risk_indirect_in, risk_indirect_out, deforested_ha, …`) and keeps extra supplier columns.
* With `filter_by_year=True`, only relations whose `years` (list, numpy array, comma-separated string or int) overlap the period are kept. Helpers: `get_years_for_period`, `filter_suppliers_by_period`.
* `supplier_risk_summary` aggregates per company, counting **unique farms** (a relation repeated per year is counted once).

### `calculate_alert(alert_indirect_df, entity_df, provider_df, n_workers=2)` — `ganabosques_risk_package.entity_alert`

Aggregates plot alerts to any entity using a plot → entity mapping (`provider_df`: `plot_id, entity_id`). Accepts the output of `total_risk` or the legacy `plot_alert_indirect` format (see the migration table below).

Output: `entity_id, entity_name, plots_total, plots_alert_direct, plots_alert_in, plots_alert_out, deforested_area_sum, alert`.

```python
from ganabosques_risk_package.entity_alert import calculate_alert

entity = pd.DataFrame({"id": ["X", "Y"], "name": ["Region A", "Region B"]})
provider = pd.DataFrame({"plot_id": [101, 102, 103], "entity_id": ["X", "X", "Y"]})
df_entity = calculate_alert(risk, entity, provider)
```

---

## ⚠️ Deprecated modules and migration

`plot_alert_direct`, `plot_alert_direct_serial` and `plot_alert_indirect` still work but emit a `DeprecationWarning` and will be removed in a future version.

| Legacy | New |
|---|---|
| `plot_alert_direct.alert_direct(plots, raster, protected, farming)` | `spatial_metrics(...)` + `alert_direct(plots, raster, metrics_df=...)` |
| `plot_alert_indirect.alert_indirect(direct_df, movement_df)` | `alert_indirect(direct_df, movements_df)` |
| `alert_direct`, `deforested_area`, `plot_area` | `direct_alert`, `deforested_ha`, `total_ha` |
| `alert_in` (plot **sends to** an alerted plot) | `indirect_alert_out` |
| `alert_out` (plot **receives from** an alerted plot) | `indirect_alert_in` |

> **Note:** "in" and "out" have **opposite meanings** in the two APIs. `calculate_alert` translates them automatically, so its output columns keep their legacy meaning.

---

## 📁 Package structure

```
ganabosques_risk_package/
├── spatial_metrics.py      # Frontier / protected-area metrics per plot
├── alert_direct.py         # Direct alerts (deforestation raster)
├── alert_indirect.py       # Indirect alerts (movement graph)
├── total_risk.py           # Consolidated risk per plot
├── alert_enterprise.py     # Company alerts via movements
├── supplier_risk.py        # Company risk via supplier relations
├── entity_alert.py         # Aggregation to any entity
├── _utils.py               # Shared helpers (ID normalisation, booleans, output)
└── plot_alert_*.py         # Deprecated legacy implementations
tests/                      # unittest suite + fixtures in tests/data_test
```

---

## ⚙️ Developer guide

```bash
pip install -r requirements.txt     # exact environment used in CI
python -m unittest discover -s tests -t . -v
```

CI (`.github/workflows/ganabosques_risk_package.yml`) runs the full suite on pushes and pull requests to `develop`, `stage` and `main`. Only pushes to `stage`, `main` and `v*` tags merge into `main` and publish a release.

On Windows, the legacy functions with `n_workers > 1` use multiprocessing, so scripts calling them must be protected with `if __name__ == "__main__":`.

---

## 📝 License and citation

Authors: Alliance Bioversity International & CIAT (Steven Sotelo and team)
Repository: https://github.com/CIAT-DAPA/ganabosques_risk_package
License: MIT

If you use this package, please cite:

Sotelo, S. (2025). Ganabosques Risk Package: Alert computation for monitoring deforestation and environmental indicators. Alliance Bioversity International & CIAT.
