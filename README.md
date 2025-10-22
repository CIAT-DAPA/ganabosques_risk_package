# Ganabosques Risk Package

The **Ganabosques Risk Package** provides a complete analytical workflow to compute **deforestation alerts** at the **plot level** and aggregate them to organizational or administrative entities.  
It includes three main analytical stages:

1. **`alert_direct`** – Calculates **direct deforestation alerts** per plot (based on deforestation rasters, protected areas, and farming areas).  
2. **`alert_indirect`** – Propagates **indirect alerts** through movements or connections between plots (origin/destination relationships).  
3. **`calculate_alert`** – Aggregates all plot-level alerts into **any entity** (e.g., ADM3, associations, providers) using a mapping DataFrame (`provider`).

This package supports **parallelization** (via `ProcessPoolExecutor`) and displays **progress bars** (`tqdm`), allowing it to scale efficiently for large spatial datasets.

---

## 📦 Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/CIAT-DAPA/ganabosques_risk_package.git
```

## 🔄 Upgrade to the latest version
```bash
pip install --upgrade git+https://github.com/CIAT-DAPA/ganabosques_risk_package.git
```

## ❌ Uninstall
```bash
pip uninstall ganabosques_risk_package
```

# 🧠 Usage and public functions

## 1️⃣ alert_direct

**Module:** ganabosques_risk_package.plot_alert_direct

**Description**
Calculates per-plot intersection metrics between the deforestation raster and vector layers of protected and farming areas.
Reports deforested areas (ha), proportions, and a boolean flag alert_direct (True if deforestation pixels are detected inside the plot).

**Function signature**

```python
alert_direct(
    plots: gpd.GeoDataFrame,
    deforestation: str,
    protected_areas: str,
    farming_areas: str,
    deforestation_value: float = 2,
    n_workers: int = 2,
    id_column: str = "id",
) -> pd.DataFrame
```

**Key parameters**

* plots: GeoDataFrame of polygons representing plots.
* deforestation: Path to the deforestation raster (GeoTIFF).
* protected_areas: Path to a shapefile or GeoJSON of protected areas.
* farming_areas: Path to a shapefile or GeoJSON of farming areas.
* n_workers: Number of processes to use (parallelization).
* id_column: Column name containing plot IDs (default "id").

**Output DataFrame columns**

* id, plot_area
* deforested_area, deforested_proportion
* protected_areas_area, protected_areas_proportion
* farming_in_area, farming_in_proportion
* farming_out_area, farming_out_proportion
* alert_direct (boolean flag)

**Example**

```python
from ganabosques_risk_package.plot_alert_direct import alert_direct
import geopandas as gpd

plots = gpd.read_file("data/plots.shp")
df_direct = alert_direct(
    plots=plots,
    deforestation="data/deforestation.tif",
    protected_areas="data/protected_areas.shp",
    farming_areas="data/farming_areas.shp",
    deforestation_value=2,
    n_workers=4,
)
print(df_direct.head())
```

## 2️⃣ alert_indirect

**Module:** ganabosques_risk_package.plot_alert_indirect

**Description**
Computes indirect alerts based on plot-to-plot interactions or movements (movement_df), assigning:

* alert_in: True if a plot sends activity to a destination that has a direct alert.
* alert_out: True if a plot receives activity from an origin that has a direct alert.

**Function signature**

```python
alert_indirect(
    alert_direct_df: pd.DataFrame,
    movement_df: pd.DataFrame,
    n_workers: int = 2,
) -> pd.DataFrame
```

**Key parameters**

* alert_direct_df: DataFrame output from alert_direct(...), containing at least id and alert_direct.
* movement_df: DataFrame describing plot-to-plot connections with:
* origen_id
* destination_id
* n_workers: Number of processes to use for parallel computation (default 2).

**Output DataFrame columns**

* Same as alert_direct_df, plus:
* alert_in (bool)
* alert_out (bool)

**Example**

```python
from ganabosques_risk_package.plot_alert_indirect import alert_indirect
import pandas as pd

df_direct = pd.read_parquet("outputs/alert_direct.parquet")
movement = pd.read_parquet("data/movement.parquet")

df_indirect = alert_indirect(df_direct, movement, n_workers=4)
print(df_indirect[["id", "alert_direct", "alert_in", "alert_out"]].head())
```

## 3️⃣ calculate_alert

**Module:** ganabosques_risk_package.entity_alert

**Description**

Aggregates plot-level alerts into any entity (e.g., administrative division, organization, or provider).
Uses a mapping DataFrame (provider) that links plots (plot_id) to entities (entity_id).
Calculates the total number of plots, the number of plots with direct and indirect alerts, total deforested area, and a final boolean flag alert indicating whether any plot has an alert.

**Function signature**

```python
calculate_alert(
    alert_indirect_df: pd.DataFrame,
    entity_df: pd.DataFrame,
    provider_df: pd.DataFrame,
    n_workers: int = 2,
) -> pd.DataFrame
```

**Key parameters**

* alert_indirect_df: Output of alert_indirect with columns id, deforested_area, alert_direct, alert_in, alert_out.
* entity_df: Master DataFrame of entities, normalized to entity_id and entity_name (accepts variants like id, name).
* provider_df: Mapping between plots and entities (plot_id, entity_id). Duplicates are automatically removed.
* n_workers: Number of parallel processes to use (default 2).

**Output DataFrame columns**

* entity_id, entity_name
* plots_total
* plots_alert_direct
* plots_alert_in
* plots_alert_out
* deforested_area_sum
* alert (boolean flag)

**Example**

```python
from ganabosques_risk_package.entity_alert import calculate_alert
import pandas as pd

df_indirect = pd.read_parquet("outputs/alert_indirect.parquet")
entity = pd.DataFrame({
    "id": ["X", "Y", "Z", "W"],
    "name": ["Region A", "Region B", "Region C", "No Plots"]
})
provider = pd.DataFrame({
    "plot_id": [101, 102, 103, 104, 105],
    "entity_id": ["X", "X", "Y", "Y", "Z"]
})

df_entity = calculate_alert(df_indirect, entity, provider, n_workers=4)
print(df_entity)
```

# 📁 Package Structure

```bash
ganabosques_risk_package/
│
├── plot_alert_direct.py      # Direct deforestation alerts (raster + vector analysis)
├── plot_alert_indirect.py    # Indirect alerts from origin/destination movements
├── entity_alert.py           # Generic entity-level aggregation (provider mapping)
└── tests/                    # Unit tests (unittest)
```

# ⚙️ Developer Guide

## 🧪 Run Unit Tests
Run all test cases in the tests/ directory:

```bash
python -m unittest discover -s tests -v
```

Run a specific test file:

```bash
python -m unittest tests.test_plot_alert_direct -v
python -m unittest tests.test_plot_alert_indirect -v
python -m unittest tests.test_entity_alert -v
```

# Environment requirements:

geopandas, rasterio, shapely, numpy, pandas, tqdm
We recommend using pip for installing geospatial dependencies.

# 📝 License and Citation
Authors: Alliance Bioversity International & CIAT (Steven Sotelo and Team)
Repository: https://github.com/CIAT-DAPA/ganabosques_risk_package
License: MIT

If you use this package, please cite:

Sotelo, S. (2025). Ganabosques Risk Package: Alert computation for monitoring deforestation and enviromental indicators.
Alliance Bioversity International & CIAT.