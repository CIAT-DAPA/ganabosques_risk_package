from enum import Enum

class RiskLevel(Enum):
    """
    Enum: RiskLevel
    Description:
        Represents the categorical levels of direct deforestation risk
        according to the MRV protocol used in the Ganabosques system.

    Values:
        - LOW: Low risk of deforestation.
        - MEDIUM: Medium risk of deforestation.
        - HIGH: High risk of deforestation.
        - NO_RISK: No risk of deforestation.
    """
    LOW = "BAJO"
    MEDIUM = "MEDIO"
    HIGH = "ALTO"
    NO_RISK = "SIN RIESGO"