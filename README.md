# Ganabosques Risk Package

This package implements the risk classification methodology from the MRV protocol for the Ganabosques system.

## Installation

```bash
pip install git+https://github.com/your-org/ganabosques_risk_package.git
```

## Modules

- `calculate_risk_direct(df_predios, raster_deforestacion, shp_areas_protegidas, shp_frontera_agro)`: Calculate direct risk per property.
- `calculate_risk_movement(predio_id, movilizaciones_df)`: Calculate entry and exit movement risks.
- `calculate_risk_total(df_predios)`: Calculate total risk from direct and movement risks.
- `calculate_risk_enterprise(empresas_df, proveedores_df)`: Aggregate risk per enterprise.
- `calculate_risk_administrative(predios_df)`: Aggregate risk by administrative units (e.g., vereda).

## Example Usage

```python
from ganabosques_risk_package.calculate_risk_direct import calculate_risk_direct

# Calculate direct risk (example)
result = calculate_risk_direct(
    df_predios=my_dataframe,
    raster_deforestacion="deforestacion.tif",
    shp_areas_protegidas="areas_protegidas.shp",
    shp_frontera_agro="frontera_agro.shp"
)
```

## Testing

```bash
python -m unittest discover tests
```
