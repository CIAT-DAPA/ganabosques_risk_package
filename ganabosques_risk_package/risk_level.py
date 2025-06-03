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
    HIGH = 3
    MEDIUM = 2
    LOW = 1
    NO_RISK = 0

def classify_risk(score):
    """
    Classify a numeric risk score into one of the RiskLevel categories.
    """
    if score >= 2.5:
        return RiskLevel.HIGH
    elif score >= 1.5:
        return RiskLevel.MEDIUM
    elif score > 0:
        return RiskLevel.LOW
    else:
        return RiskLevel.NO_RISK