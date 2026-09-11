"""Guilloché / Background pattern detector wrapper.

Evaluates document background security patterns against authentic reference patterns.
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
    """Inference wrapper for Guilloché and background security pattern verification."""

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
        """Inspect background pattern of document against authentic reference.

        Args:
            image: Document PIL Image.
            doc_type: Document type name.
            country: Issuing country identifier (e.g. USA, IND, GBR).
            version: Series/version of document.
            region: Specific pattern region to verify.
            is_applicable: Set to False if document type does not possess guilloche patterns.

        Returns:
            Structured dictionary with Guilloché verification result.
        """
        if not is_applicable:
            logger.info(f"Guilloché inspection for {doc_type}: NOT_APPLICABLE")
            return {
                "status": GuillocheStatus.NOT_APPLICABLE.value,
                "passed": None,
                "similarity_score": None,
                "threshold": self.similarity_threshold,
                "reference_available": False,
                "reason": "Feature is not applicable to this document type/region",
            }

        # Retrieve reference pattern
        reference_pattern = self.reference_manager.get_reference(
            country=country,
            document_type=doc_type,
            version=version,
            region=region,
        )

        if reference_pattern is None:
            logger.info(f"Guilloché inspection for {doc_type}: REFERENCE_REQUIRED (no reference found)")
            return {
                "status": GuillocheStatus.REFERENCE_REQUIRED.value,
                "passed": None,
                "similarity_score": None,
                "threshold": self.similarity_threshold,
                "reference_available": False,
                "reason": f"No authentic reference pattern available for {country}/{doc_type}",
            }

        try:
            # Extract pattern region from query document
            query_pattern = self.pattern_extractor.extract_pattern(image, region_type=region or "background")

            # Calculate similarity
            sim = self.similarity_evaluator.compute_similarity(query_pattern, reference_pattern)
            is_consistent = sim >= self.similarity_threshold
            status = (
                GuillocheStatus.CONSISTENT.value
                if is_consistent
                else GuillocheStatus.INCONSISTENT.value
            )

            logger.info(
                f"Guilloché inspection for {doc_type}: {status} "
                f"(similarity={sim:.4f}, threshold={self.similarity_threshold})"
            )

            return {
                "status": status,
                "passed": is_consistent,
                "similarity_score": round(sim, 4),
                "threshold": self.similarity_threshold,
                "reference_available": True,
                "reason": None,
            }

        except Exception as e:
            logger.error(f"Error inspecting guilloche pattern for {doc_type}: {e}")
            return {
                "status": GuillocheStatus.ERROR.value,
                "passed": False,
                "similarity_score": None,
                "threshold": self.similarity_threshold,
                "reference_available": True,
                "reason": str(e),
            }
