"""shadowselect: shadow-feature selection that is honest about class imbalance."""

from .diagnostics import (
    ImbalanceReport,
    ImbalanceWarning,
    LearnerWarning,
    assess_imbalance,
    minority_count,
    null_keep_rate,
)
from .screening import UnivariateScreen, benjamini_hochberg
from .selector import ShadowFeatureSelector

__all__ = [
    "ShadowFeatureSelector",
    "UnivariateScreen",
    "benjamini_hochberg",
    "ImbalanceWarning",
    "LearnerWarning",
    "ImbalanceReport",
    "assess_imbalance",
    "minority_count",
    "null_keep_rate",
]

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("shadowselect")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0+unknown"
