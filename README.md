# Ganabosques Risk Package

This package implements the risk classification methodology from the MRV protocol for the Ganabosques system.

## Installation

```bash
pip install git+https://github.com/your-org/ganabosques_risk_package.git
```

## Modules

- `calculate_risk_direct(df_plots, raster_deforestation, shp_protected_areas, shp_farming_areas)`: Calculate direct risk per property.
- `calculate_risk_movement(plot_id, movilizaciones_df)`: Calculate entry and exit movement risks.
- `calculate_risk_total(df_plots)`: Calculate total risk from direct and movement risks.
- `calculate_risk_enterprise(empresas_df, proveedores_df)`: Aggregate risk per enterprise.
- `calculate_risk_administrative(predios_df)`: Aggregate risk by administrative units (e.g., vereda).

## Example Usage

```python
from ganabosques_risk_package.calculate_risk_direct import calculate_risk_direct

# Calculate direct risk (example)
result = calculate_risk_direct(
    df_plots=my_dataframe,
    raster_deforestation="deforestation.tif",
    shp_protected_areas="protected_areas.shp",
    shp_farming_areas="farming_areas.shp"
)
```

## Testing

```bash
python -m unittest discover tests
```
