"""Output schemas for Module 2 / Risk Engine."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CaseStatus(str, Enum):
    RED_FLAG = "RED_FLAG"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class AIGateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PASSED = "PASS"
    FAILED = "FAIL"
    SKIPPED = "SKIPPED"


class TamperStatus(str, Enum):
    BONA_FIDE = "BONA_FIDE"
    FORGED = "FORGED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class GuillocheStatus(str, Enum):
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REFERENCE_REQUIRED = "REFERENCE_REQUIRED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class DocumentForensicResult:
    document_type: str
    ai_test: str = "PASS"
    ai_gate_status: str = "PASS"
    ai_generated: bool = False
    tamper_result: Optional[Dict[str, Any]] = None
    guilloche_result: Optional[Dict[str, Any]] = None
    forensic_result: Optional[Dict[str, Any]] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class CaseOutput:
    uuid: str
    status: str
    ai_test: str = "PASS"
    ai_generation_check: str = "PASS"
    forensic_pass_score: Optional[float] = None
    forgery_score: Optional[float] = None
    weights_applied: Optional[Dict[str, Any]] = None
    points_of_failure: List[Dict[str, Any]] = field(default_factory=list)
    flag_type: Optional[str] = None
    flagged_document: Optional[str] = None
    documents_analyzed: List[str] = field(default_factory=list)
    document_results: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self):
        data = {
            "uuid": self.uuid,
            "status": self.status,
            "ai_test": self.ai_test,
            "ai_generation_check": self.ai_test,
            "forensic_pass_score": self.forensic_pass_score,
            "forgery_score": self.forgery_score if self.forgery_score is not None else self.forensic_pass_score,
            "weights_applied": self.weights_applied or {},
            "points_of_failure": self.points_of_failure,
            "failures_detected": len(self.points_of_failure),
        }
        if self.status == CaseStatus.RED_FLAG.value:
            data["flag_type"] = self.flag_type or "FORENSIC_FAILURE"
            data["flagged_document"] = self.flagged_document
        if self.documents_analyzed:
            data["documents_analyzed"] = self.documents_analyzed
        if self.document_results:
            data["document_results"] = self.document_results
        if self.error_message:
            data["error_message"] = self.error_message
        return data
