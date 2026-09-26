"""Guilloché / Security-Pattern production detector.

The common Module 2 input gate runs before this branch. This detector performs
only Guilloché-specific preprocessing: pattern extraction -> crop -> resize ->
Siamese ResNet-18 -> cosine similarity against an authentic reference.
"""

from typing import Any, Dict, Optional
from PIL import Image

from module2.guilloche.pattern_extractor import PatternExtractor
from module2.guilloche.reference_manager import GuillocheReferenceManager
from module2.guilloche.similarity import GuillocheSimilarityEvaluator
from module2.schemas.output_schema import GuillocheStatus
from module2.utils.logger import get_logger

logger = get_logger("module2.guilloche.guilloche_detector")


class GuillocheDetector:
    def __init__(
        self,
        reference_manager: Optional[GuillocheReferenceManager] = None,
        similarity_evaluator: Optional[GuillocheSimilarityEvaluator] = None,
        pattern_extractor: Optional[PatternExtractor] = None,
        similarity_threshold: float = 0.75,
    ):
        self.reference_manager = reference_manager or GuillocheReferenceManager()
        self.similarity_evaluator = similarity_evaluator or GuillocheSimilarityEvaluator()
        self.pattern_extractor = pattern_extractor or PatternExtractor()
        self.similarity_threshold = similarity_threshold

    def inspect_pattern(
        self,
        image: Image.Image,
        doc_type: str,
        country: Optional[str] = None,
        version: Optional[str] = "default",
        region: Optional[str] = "background",
        is_applicable: bool = True,
    ) -> Dict[str, Any]:
        if not is_applicable:
            return {
                "status": GuillocheStatus.NOT_APPLICABLE.value,
                "passed": None,
                "similarity_score": None,
                "threshold": self.similarity_threshold,
                "reference_available": False,
                "failure_reasons": [],
            }

        reference_pattern = self.reference_manager.get_reference(
            country=country,
            document_type=doc_type,
            version=version,
            region=region,
        )

        if reference_pattern is None:
            return {
                "status": GuillocheStatus.REFERENCE_REQUIRED.value,
                "passed": None,
                "similarity_score": None,
                "threshold": self.similarity_threshold,
                "reference_available": False,
                "failure_reasons": [],
                "reason": "REFERENCE_NOT_AVAILABLE",
            }

        try:
            query_pattern = self.pattern_extractor.extract_pattern(
                image,
                region_type=region or "background",
            )
            sim = self.similarity_evaluator.compute_similarity(
                query_pattern,
                reference_pattern,
            )
            consistent = sim >= self.similarity_threshold

            return {
                "status": (
                    GuillocheStatus.CONSISTENT.value
                    if consistent
                    else GuillocheStatus.INCONSISTENT.value
                ),
                "passed": consistent,
                "similarity_score": round(float(sim), 4),
                "threshold": self.similarity_threshold,
                "reference_available": True,
                "failure_reasons": (
                    [] if consistent else ["GUILLOCHE_PATTERN_INCONSISTENT"]
                ),
            }
        except Exception as e:
            logger.error("Guilloché analysis failed for %s: %s", doc_type, e)
            return {
                "status": GuillocheStatus.ERROR.value,
                "passed": False,
                "similarity_score": None,
                "threshold": self.similarity_threshold,
                "reference_available": True,
                "failure_reasons": [],
                "error": str(e),
            }
