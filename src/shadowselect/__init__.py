"""shadowselect: shadow-feature selection that is honest about class imbalance."""

from .diagnostics import ImbalanceReport, ImbalanceWarning, assess_imbalance, minority_count
from .selector import ShadowFeatureSelector

__all__ = [
    "ShadowFeatureSelector",
    "ImbalanceWarning",
    "ImbalanceReport",
    "assess_imbalance",
    "minority_count",
]
__version__ = "0.1.0"
