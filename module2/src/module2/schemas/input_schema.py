"""Input schemas and validation for Module 2.

Handles case-level inputs, required/optional documents, and formats.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import numpy as np


class DocumentType(str, Enum):
    """Supported document types in deterministic processing order."""
    PASSPORT = "passport"
    VISA = "visa"
    NATIONAL_ID = "national_id"
    DRIVING_LICENSE = "driving_license"
    PERMIT = "permit"


REQUIRED_DOCUMENT_TYPES = {
    DocumentType.PASSPORT.value,
    DocumentType.VISA.value,
    DocumentType.NATIONAL_ID.value,
}

OPTIONAL_DOCUMENT_TYPES = {
    DocumentType.DRIVING_LICENSE.value,
    DocumentType.PERMIT.value,
}

ALL_DOCUMENT_TYPES = REQUIRED_DOCUMENT_TYPES | OPTIONAL_DOCUMENT_TYPES

# Image input can be a filesystem path, PIL Image, or numpy array
ImageInputType = Union[str, Path, Image.Image, np.ndarray, bytes]


class ValidationError(Exception):
    """Raised when input validation fails."""
    pass


@dataclass
class CaseInput:
    """Represents a validated case-level input for Module 2."""
    uuid: str
    documents_present: List[str]
    passport_image: Optional[ImageInputType] = None
    visa_image: Optional[ImageInputType] = None
    national_id_image: Optional[ImageInputType] = None
    driving_license_image: Optional[ImageInputType] = None
    permit_image: Optional[ImageInputType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate case input according to Module 2 specification.

        Raises:
            ValidationError: If required documents are missing or declared invalidly.
        """
        if not self.uuid or not isinstance(self.uuid, str):
            raise ValidationError("Case uuid must be a non-empty string.")

        if not self.documents_present or not isinstance(self.documents_present, list):
            raise ValidationError("documents_present must be a non-empty list of strings.")

        # Normalize document names
        normalized_present = [d.strip().lower() for d in self.documents_present]

        # Check for unrecognized document types
        for doc in normalized_present:
            if doc not in ALL_DOCUMENT_TYPES:
                raise ValidationError(
                    f"Unrecognized document type '{doc}' in documents_present. "
                    f"Must be one of {sorted(ALL_DOCUMENT_TYPES)}"
                )

        # Check that all required documents are present
        missing_required = REQUIRED_DOCUMENT_TYPES - set(normalized_present)
        if missing_required:
            raise ValidationError(
                f"Missing compulsory documents in documents_present: {sorted(missing_required)}. "
                f"Compulsory documents are: {sorted(REQUIRED_DOCUMENT_TYPES)}"
            )

        # Check total count: valid cases have 3, 4, or 5 documents
        if len(normalized_present) not in (3, 4, 5):
            raise ValidationError(
                f"Invalid number of documents in documents_present: {len(normalized_present)}. "
                "Allowed: 3 (passport, visa, national_id), 4 (plus driving_license or permit), or 5 (all)."
            )

        # Check image availability for each declared document
        for doc in normalized_present:
            img = self.get_image(doc)
            if img is None:
                raise ValidationError(
                    f"Document '{doc}' is listed in documents_present, but {doc}_image is None or missing."
                )

    def get_image(self, doc_type: str) -> Optional[ImageInputType]:
        """Retrieve the image input associated with a document type."""
        attr_name = f"{doc_type.strip().lower()}_image"
        return getattr(self, attr_name, None)
