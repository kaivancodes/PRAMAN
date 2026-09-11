"""Models package for Module 2."""

from module2.models.efficientnet_detector import (
    CLASS_NAMES,
    SIDTDEfficientNetDetector,
    build_efficientnet_b3,
)

__all__ = ["build_efficientnet_b3", "SIDTDEfficientNetDetector", "CLASS_NAMES"]
