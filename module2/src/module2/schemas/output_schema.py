"""Output schemas and results serialization for Module 2.

Provides clean contract for Risk Engine and structured forensic details.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CaseStatus(str, Enum):
    """Overall Case status returned by Module 2."""
    RED_FLAG = "RED_FLAG"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class AIGateStatus(str, Enum):
    """Status of the Module 2.1 AI-generation gate."""
    PASS = "PASS"
    FAIL = "FAIL"
    PASSED = "PASS"  # alias for backwards compatibility
    FAILED = "FAIL"  # alias for backwards compatibility
    SKIPPED = "SKIPPED"


class TamperStatus(str, Enum):
    """Status returned by Module 2.2 Tamper / Forgery branch."""
    BONA_FIDE = "BONA_FIDE"
    FORGED = "FORGED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class GuillocheStatus(str, Enum):
    """Status returned by Module 2.2 Guilloché / Pattern branch."""
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REFERENCE_REQUIRED = "REFERENCE_REQUIRED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class DocumentForensicResult:
    """Document-level internal results across Module 2.1 and Module 2.2."""
    document_type: str
    ai_test: str = AIGateStatus.PASS.value  # "PASS" | "FAIL"
    ai_gate_status: str = AIGateStatus.PASS.value  # "PASS" | "FAIL"
    ai_generated: bool = False
    tamper_result: Optional[Dict[str, Any]] = None
    guilloche_result: Optional[Dict[str, Any]] = None
    forensic_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaseOutput:
    """Risk Engine contract and final case output for Module 2."""
    uuid: str
    status: str  # "RED_FLAG" | "COMPLETED" | "ERROR"
    ai_test: str = AIGateStatus.PASS.value  # "PASS" | "FAIL"
    ai_generation_check: str = AIGateStatus.PASS.value  # "PASS" | "FAIL"
    forensic_pass_score: Optional[float] = None
    weights_applied: Optional[Dict[str, Any]] = None
    points_of_failure: List[Dict[str, Any]] = field(default_factory=list)
    flag_type: Optional[str] = None  # "AI_GENERATED_DOCUMENT" if RED_FLAG
    flagged_document: Optional[str] = None  # document type that triggered RED_FLAG
    documents_analyzed: List[str] = field(default_factory=list)
    document_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching exact specification contract."""
        data: Dict[str, Any] = {
            "uuid": self.uuid,
            "status": self.status,
            "ai_test": self.ai_test,
            "ai_generation_check": self.ai_test,
            "forensic_pass_score": self.forensic_pass_score,
            "weights_applied": self.weights_applied or {},
            "points_of_failure": self.points_of_failure,
            "failures_detected": len(self.points_of_failure),
        }

        if self.status == CaseStatus.RED_FLAG.value:
            data["flag_type"] = self.flag_type or "AI_GENERATED_DOCUMENT"
            data["flagged_document"] = self.flagged_document

        if self.documents_analyzed:
            data["documents_analyzed"] = self.documents_analyzed

        if self.document_results:
            data["document_results"] = self.document_results

        if self.error_message:
            data["error_message"] = self.error_message

        return data


