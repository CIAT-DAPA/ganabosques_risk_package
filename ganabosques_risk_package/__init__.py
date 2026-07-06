# -*- coding: utf-8 -*-
"""
ganabosques_risk_package
========================

Paquete reutilizable para cálculo de riesgo de deforestación asociado a
fincas ganaderas. Recibe y retorna DataFrames.

Módulos
-------
- **spatial_metrics** : Métricas espaciales (área en frontera, protegida).
- **alert_direct**    : Alerta directa por intersección con raster de deforestación.
- **alert_indirect**  : Alerta indirecta por grafos de movimiento.
- **total_risk**      : Consolidación de riesgo total por predio.
- **alert_enterprise**: Alertas de empresa por movimientos de ganado.
- **supplier_risk**   : Riesgo de empresa por relaciones supplier.

Flujo típico
-------------
.. code-block:: python

    from ganabosques_risk_package import (
        spatial_metrics,
        alert_direct,
        alert_indirect,
        total_risk,
        alert_enterprise,
        supplier_risk,
    )

    # 1. Métricas espaciales (una vez)
    metrics = spatial_metrics(plots, farming_areas, protected_areas)

    # 2. Alerta directa (por cada capa de deforestación)
    direct = alert_direct(plots, "deforestation.tif", metrics_df=metrics)

    # 3. Alerta indirecta (movimientos)
    indirect = alert_indirect(direct, movements)

    # 4. Riesgo total consolidado
    risk = total_risk(direct, indirect, metrics)

    # 5. Alertas de empresa (movimientos)
    ent_alerts = alert_enterprise(risk, movements)

    # 6. Riesgo de empresa por suppliers
    sup_risk = supplier_risk(risk, suppliers_df)

Autor: CIAT-DAPA / Ganabosques
"""

# ---- Public API ----

from .spatial_metrics import spatial_metrics
from .alert_direct import alert_direct
from .alert_indirect import alert_indirect
from .total_risk import total_risk
from .alert_enterprise import alert_enterprise
from .supplier_risk import supplier_risk, supplier_risk_summary
from .supplier_risk import get_years_for_period, filter_suppliers_by_period

__all__ = [
    "spatial_metrics",
    "alert_direct",
    "alert_indirect",
    "total_risk",
    "alert_enterprise",
    "supplier_risk",
    "supplier_risk_summary",
    "get_years_for_period",
    "filter_suppliers_by_period",
]
