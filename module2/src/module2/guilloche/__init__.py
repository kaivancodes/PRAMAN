"""Guilloché / Background pattern forensics package."""

from module2.guilloche.guilloche_detector import GuillocheDetector
from module2.guilloche.pattern_extractor import PatternExtractor
from module2.guilloche.reference_manager import GuillocheReferenceManager
from module2.guilloche.similarity import (
    GuillocheSimilarityEvaluator,
    PatternCNNEncoder,
)

__all__ = [
    "GuillocheDetector",
    "PatternExtractor",
    "GuillocheReferenceManager",
    "GuillocheSimilarityEvaluator",
    "PatternCNNEncoder",
]
