"""Schemas package for Module 2."""

from module2.schemas.input_schema import (
    ALL_DOCUMENT_TYPES,
    CaseInput,
    DocumentType,
    OPTIONAL_DOCUMENT_TYPES,
    REQUIRED_DOCUMENT_TYPES,
    ValidationError,
)
from module2.schemas.output_schema import (
    AIGateStatus,
    CaseOutput,
    CaseStatus,
    DocumentForensicResult,
    GuillocheStatus,
    TamperStatus,
)

__all__ = [
    "ALL_DOCUMENT_TYPES",
    "CaseInput",
    "DocumentType",
    "OPTIONAL_DOCUMENT_TYPES",
    "REQUIRED_DOCUMENT_TYPES",
    "ValidationError",
    "AIGateStatus",
    "CaseOutput",
    "CaseStatus",
    "DocumentForensicResult",
    "GuillocheStatus",
    "TamperStatus",
]
