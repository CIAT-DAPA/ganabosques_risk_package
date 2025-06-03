
# Ganabosques Risk Package

This package implements the risk classification methodology from the MRV protocol for the Ganabosques system.

## Installation

```bash
pip install git+https://github.com/your-org/ganabosques_risk_package.git
```

## Modules & Usage Examples

### 1. `calculate_risk_direct`

Calculates the direct deforestation risk per plot using geospatial buffers, protected areas, deforestation raster, and agro-frontier maps.

```python
from ganabosques_risk_package.calculate_risk_direct import calculate_risk_direct

direct_df = calculate_risk_direct(
    df_plots=my_geodataframe,  # GeoDataFrame with 'geometry' and 'id'
    raster_deforestation_path="data/deforestation.tif",
    shp_protected_areas_path="data/protected_areas.shp",
    shp_farming_areas_path="data/farming_frontier.shp"
)
```

### 2. `calculate_risk_movement`

Computes average entry and exit risk levels based on movement relationships between plots.

```python
from ganabosques_risk_package.calculate_risk_movement import calculate_risk_movement

movement_df = calculate_risk_movement(
    df_plots_risk=direct_df,  # Output of calculate_risk_direct
    df_movement=movements_df  # DataFrame with 'origen_id' and 'destination_id'
)
```

### 3. `calculate_risk_total`

Calculates the total risk per plot by combining direct, entry, and exit risks with defined weights.

```python
from ganabosques_risk_package.calculate_risk_total import calculate_risk_total

total_df = calculate_risk_total(
    df_direct=direct_df,      # Output from calculate_risk_direct
    df_movement=movement_df   # Output from calculate_risk_movement
)
```

### 4. `calculate_risk_enterprise`

Computes the enterprise-level risk using supplier relationships and aggregated plot risks.

```python
from ganabosques_risk_package.calculate_risk_enterprise import calculate_risk_enterprise

enterprise_df = calculate_risk_enterprise(
    df_enterprises=enterprises_df,        # DataFrame with enterprise 'id'
    df_suppliers=supplier_df,             # DataFrame with 'enterprise_id', 'plot_id'
    df_plots_risk=total_df                # Output from calculate_risk_total
)
```

### 5. `calculate_risk_administrative`

Aggregates average risk levels across administrative levels (departments, municipalities, veredas) based on plot and enterprise data.

```python
from ganabosques_risk_package.calculate_risk_administrative import calculate_risk_administrative

admin_df = calculate_risk_administrative(
    df_plots=total_df,         # Includes administrative levels and 'risk_total'
    df_enterprises=enterprise_df  # Includes admin levels and 'enterprise_risk_enum'
)
```

## Risk Enum Utility

The system uses the following enum to classify risk levels:

```python
from ganabosques_risk_package.risk_level import RiskLevel

print(RiskLevel.HIGH.name)   # "HIGH"
print(RiskLevel.HIGH.value)  # 3
```

## Testing

To run all unit tests:

```bash
python -m unittest discover tests
```
