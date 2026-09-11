"""Forensic pass scoring logic for Module 2.2.

Calculates a score out of 100 representing the percentage of weighted forensic checks that passed.
Note: The forensic pass score is NOT an authenticity probability.
"""

from typing import Any, Dict, List, Optional
from module2.schemas.output_schema import GuillocheStatus, TamperStatus
from module2.utils.logger import get_logger

logger = get_logger("module2.scoring.forensic_scorer")


class ForensicScorer:
    """Calculates weighted forensic pass score out of 100 for documents and cases."""

    def __init__(
        self,
        tamper_weight: float = 0.60,
        guilloche_weight: float = 0.40,
        missing_ref_policy: str = "REWEIGHT",
    ):
        """Initialize ForensicScorer.

        Args:
            tamper_weight: Weight assigned to splice/tamper check (default 0.60).
            guilloche_weight: Weight assigned to guilloché/background check (default 0.40).
            missing_ref_policy: Strategy when reference is unavailable:
                                'REWEIGHT' (re-normalize by available checks),
                                'STRICT_FAIL' (treat missing reference as unpassed),
                                'NEUTRAL' (award half-weight).
        """
        total = tamper_weight + guilloche_weight
        if abs(total - 1.0) > 1e-4:
            logger.warning(
                f"Weights sum to {total}, not 1.0. Normalizing: "
                f"tamper={tamper_weight/total:.2f}, guilloche={guilloche_weight/total:.2f}"
            )
            self.tamper_weight = tamper_weight / total
            self.guilloche_weight = guilloche_weight / total
        else:
            self.tamper_weight = tamper_weight
            self.guilloche_weight = guilloche_weight

        self.missing_ref_policy = missing_ref_policy.upper()

    def get_weights(self) -> Dict[str, Any]:
        """Return the active forensic scoring weights and policy configuration."""
        return {
            "tamper_weight": round(self.tamper_weight, 4),
            "guilloche_weight": round(self.guilloche_weight, 4),
            "scoring_policy_missing_ref": self.missing_ref_policy,
        }

    def score_document(
        self,
        tamper_result: Dict[str, Any],
        guilloche_result: Dict[str, Any],
        doc_type: str = "document",
    ) -> Dict[str, Any]:
        """Calculate forensic score for an individual document.

        Formula:
          Score = (Tamper_Passed * W_tamper + Guilloche_Passed * W_guilloche) / W_available * 100

        Args:
            tamper_result: Output dictionary from TamperDetector.
            guilloche_result: Output dictionary from GuillocheDetector.
            doc_type: Name of document type.

        Returns:
            Dictionary containing doc_score, checks_evaluated, and details.
        """
        # 1. Evaluate Tamper Branch
        tamper_passed = bool(tamper_result.get("passed", False))
        tamper_status = tamper_result.get("status", TamperStatus.ERROR.value)
        tamper_score_contrib = self.tamper_weight if tamper_passed else 0.0
        applicable_weight = self.tamper_weight

        # 2. Evaluate Guilloché Branch
        guilloche_status = guilloche_result.get("status")
        guilloche_passed = guilloche_result.get("passed")

        guilloche_score_contrib = 0.0
        guilloche_weight_used = self.guilloche_weight

        if guilloche_status == GuillocheStatus.CONSISTENT.value:
            guilloche_score_contrib = self.guilloche_weight
            applicable_weight += self.guilloche_weight
        elif guilloche_status == GuillocheStatus.INCONSISTENT.value:
            guilloche_score_contrib = 0.0
            applicable_weight += self.guilloche_weight
        elif guilloche_status in (
            GuillocheStatus.REFERENCE_REQUIRED.value,
            GuillocheStatus.NOT_APPLICABLE.value,
        ):
            if self.missing_ref_policy == "STRICT_FAIL":
                applicable_weight += self.guilloche_weight
                guilloche_score_contrib = 0.0
            elif self.missing_ref_policy == "NEUTRAL":
                applicable_weight += self.guilloche_weight
                guilloche_score_contrib = self.guilloche_weight * 0.5
            else:
                # REWEIGHT: Exclude from denominator
                guilloche_weight_used = 0.0
        else:
            # Error or unknown
            applicable_weight += self.guilloche_weight
            guilloche_score_contrib = 0.0

        if applicable_weight > 0:
            doc_score = ((tamper_score_contrib + guilloche_score_contrib) / applicable_weight) * 100.0
        else:
            doc_score = 0.0

        doc_score = round(max(0.0, min(100.0, doc_score)), 2)

        return {
            "document_type": doc_type,
            "forensic_pass_score": doc_score,
            "tamper_passed": tamper_passed,
            "tamper_status": tamper_status,
            "guilloche_status": guilloche_status,
            "guilloche_passed": guilloche_passed,
            "applicable_weight": applicable_weight,
        }

    def score_case(
        self,
        document_scores: List[Dict[str, Any]],
    ) -> Optional[float]:
        """Aggregate document forensic pass scores into a single case-level score.

        Args:
            document_scores: List of document scoring summaries.

        Returns:
            Aggregated case forensic pass score out of 100, or None if no documents scored.
        """
        if not document_scores:
            return None

        total_score = sum(d["forensic_pass_score"] for d in document_scores)
        case_score = total_score / len(document_scores)
        return round(max(0.0, min(100.0, case_score)), 2)
