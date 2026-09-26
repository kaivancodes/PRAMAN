"""Final forensic score calculation for Module 2."""

from typing import Any, Dict, List, Optional
from module2.schemas.output_schema import GuillocheStatus, TamperStatus


class ForensicScorer:
    """Combines the two forensic branches using 60/40 default weights."""

    def __init__(self, tamper_weight=0.60, guilloche_weight=0.40, missing_ref_policy="REWEIGHT"):
        total = tamper_weight + guilloche_weight
        self.tamper_weight = tamper_weight / total
        self.guilloche_weight = guilloche_weight / total
        self.missing_ref_policy = missing_ref_policy.upper()

    def get_weights(self):
        return {
            "tamper_weight": self.tamper_weight,
            "guilloche_weight": self.guilloche_weight,
            "scoring_policy_missing_ref": self.missing_ref_policy,
        }

    def score_document(self, tamper_result, guilloche_result, doc_type="document"):
        tamper_status = tamper_result.get("status", TamperStatus.ERROR.value)
        guilloche_status = guilloche_result.get("status", GuillocheStatus.ERROR.value)

        # Technical errors are incomplete, not forensic failures.
        if tamper_status == TamperStatus.ERROR.value or guilloche_status == GuillocheStatus.ERROR.value:
            return {
                "document_type": doc_type,
                "forensic_pass_score": None,
                "forgery_score": None,
                "tamper_passed": False,
                "tamper_status": tamper_status,
                "guilloche_status": guilloche_status,
                "guilloche_passed": guilloche_result.get("passed"),
                "technical_error": True,
            }

        tamper_passed = bool(tamper_result.get("passed", False))
        if guilloche_status == GuillocheStatus.CONSISTENT.value:
            g_passed = True
            g_available = True
        elif guilloche_status == GuillocheStatus.INCONSISTENT.value:
            g_passed = False
            g_available = True
        elif guilloche_status in (GuillocheStatus.REFERENCE_REQUIRED.value, GuillocheStatus.NOT_APPLICABLE.value):
            if self.missing_ref_policy == "REWEIGHT":
                g_passed = None
                g_available = False
            else:
                g_passed = False
                g_available = True
        else:
            return {
                "document_type": doc_type,
                "forensic_pass_score": None,
                "forgery_score": None,
                "tamper_passed": tamper_passed,
                "tamper_status": tamper_status,
                "guilloche_status": guilloche_status,
                "guilloche_passed": None,
                "technical_error": True,
            }

        available = self.tamper_weight + (self.guilloche_weight if g_available else 0)
        score = ((self.tamper_weight if tamper_passed else 0) +
                 (self.guilloche_weight if g_passed else 0)) / available * 100

        return {
            "document_type": doc_type,
            "forensic_pass_score": round(score, 2),
            "forgery_score": round(score, 2),
            "tamper_passed": tamper_passed,
            "tamper_status": tamper_status,
            "guilloche_status": guilloche_status,
            "guilloche_passed": g_passed,
            "technical_error": False,
        }

    def score_case(self, document_scores: List[Dict[str, Any]]) -> Optional[float]:
        valid = [d["forensic_pass_score"] for d in document_scores if d.get("forensic_pass_score") is not None]
        if not valid:
            return None
        return round(sum(valid) / len(valid), 2)
