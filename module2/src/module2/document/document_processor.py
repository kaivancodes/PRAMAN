"""Generic document processor for Module 2.

Handles document-level processing for:
- Module 2.1: AI-generated image detection gate.
- Module 2.2: Visual forensics (Tamper / Forgery and Guilloché security pattern).

Document type is represented as data, not duplicated code.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple
import numpy as np
import torch
from PIL import Image

from module2.guilloche.guilloche_detector import GuillocheDetector
from module2.preprocessing.forensic_preprocessor import ForensicPreprocessor
from module2.schemas.output_schema import AIGateStatus, DocumentForensicResult
from module2.tamper.tamper_detector import TamperDetector
from module2.utils.device import get_device
from module2.utils.logger import get_logger

logger = get_logger("module2.document.document_processor")


class BaseAIDetector(ABC):
    """Abstract interface for AI-generated document detection.

    Allows plugging in future dedicated AI-generation detection models
    without modifying the orchestrator.
    """

    @abstractmethod
    def is_ai_generated(self, normalized_tensor: torch.Tensor, original_image: Image.Image) -> bool:
        """Evaluate if the given document image is AI-generated.

        Args:
            normalized_tensor: Preprocessed tensor (1, 3, H, W) specifically for AI gate.
            original_image: Preserved original PIL Image.

        Returns:
            True if AI_GENERATED = YES, False if AI_GENERATED = NO.
        """
        pass


class ModularAIDetector(BaseAIDetector):
    """Configurable AI-generated image detector implementation.

    Accepts custom inference hooks, model paths, or simulation flags for testing.
    Scientific Integrity Rule: SIDTD is NOT used here.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        custom_detector_fn: Optional[Callable[[Image.Image], bool]] = None,
    ):
        self.model_path = model_path
        self.custom_detector_fn = custom_detector_fn
        # In-memory document flag simulation (e.g. for testing specific document types)
        self._flagged_doc_types: set = set()

    def set_flagged_for_testing(self, doc_type: str) -> None:
        """Mark a document type to trigger AI_GENERATED = YES for testing."""
        self._flagged_doc_types.add(doc_type.strip().lower())

    def clear_flags_for_testing(self) -> None:
        """Clear all test flags."""
        self._flagged_doc_types.clear()

    def is_ai_generated(self, normalized_tensor: torch.Tensor, original_image: Image.Image) -> bool:
        """Determine if document is AI-generated."""
        # If a custom detector function is provided
        if self.custom_detector_fn is not None:
            return bool(self.custom_detector_fn(original_image))

        # Check metadata or custom test attributes if attached to image
        if hasattr(original_image, "_simulated_ai_generated"):
            return bool(getattr(original_image, "_simulated_ai_generated"))

        # Default production behavior: unless flagged by a detector, return False
        return False


class DocumentProcessor:
    """Generic processor handling Module 2.1 and Module 2.2 operations for any document type."""

    def __init__(
        self,
        ai_detector: Optional[BaseAIDetector] = None,
        tamper_detector: Optional[TamperDetector] = None,
        guilloche_detector: Optional[GuillocheDetector] = None,
        forensic_preprocessor: Optional[ForensicPreprocessor] = None,
        ai_input_size: int = 224,
        device: Optional[str] = "auto",
    ):
        self.device = get_device(device)
        self.ai_detector = ai_detector or ModularAIDetector()
        self.tamper_detector = tamper_detector or TamperDetector()
        self.guilloche_detector = guilloche_detector or GuillocheDetector()
        self.forensic_preprocessor = forensic_preprocessor or ForensicPreprocessor()
        self.ai_input_size = ai_input_size

        # Preprocessing normalization tensors for AI gate
        self._ai_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self._ai_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)

    def preprocess_for_ai_gate(self, original_image: Image.Image) -> torch.Tensor:
        """Module 2.1-only preprocessing.

        Original Image -> Resize/Crop -> Tensor -> Normalization.
        Does NOT alter original_image or share normalized tensors with Module 2.2.
        """
        resized = original_image.convert("RGB").resize(
            (self.ai_input_size, self.ai_input_size), Image.Resampling.BILINEAR
        )
        arr = np.array(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).to(self.device)
        norm_tensor = (tensor - self._ai_mean) / self._ai_std
        return norm_tensor.unsqueeze(0)

    def run_ai_gate(self, original_image: Image.Image, doc_type: str) -> bool:
        """Execute Module 2.1 AI-generation gate on a single document.

        Args:
            original_image: Clean original PIL Image.
            doc_type: Document type name.

        Returns:
            True if AI Test FAILS (synthetic / AI-generated document detected).
            False if AI Test PASSES (authentic document).
        """
        logger.info(f"AI test started for {doc_type}")

        # Check if modular detector has explicit test flag for this doc_type
        if isinstance(self.ai_detector, ModularAIDetector):
            if doc_type.strip().lower() in self.ai_detector._flagged_doc_types:
                logger.warning(f"AI test flagged {doc_type}: AI_Test = FAIL (test flag)")
                return True

        ai_tensor = self.preprocess_for_ai_gate(original_image)
        is_ai = self.ai_detector.is_ai_generated(ai_tensor, original_image)

        result_str = "FAIL" if is_ai else "PASS"
        logger.info(f"{doc_type.capitalize()} AI Test result: {result_str}")
        return is_ai


    def run_forensics(
        self,
        original_image: Image.Image,
        doc_type: str,
        country: Optional[str] = None,
        version: Optional[str] = "default",
        region: Optional[str] = "background",
        is_guilloche_applicable: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Execute Module 2.2 dual-branch visual forensics.

        Args:
            original_image: Preserved original PIL Image.
            doc_type: Document type name.
            country: Optional document issuing country for Guilloché reference lookup.
            version: Optional series/version.
            region: Optional pattern region.
            is_guilloche_applicable: Whether Guilloché checks apply to this document.

        Returns:
            Tuple of (tamper_result, guilloche_result, forensic_context).
        """
        logger.debug(f"Starting Module 2.2 forensic analysis for {doc_type}")

        # 1. Independent forensic preprocessing (ELA and DCT)
        forensic_context = self.forensic_preprocessor.extract_forensic_package(
            original_image=original_image,
            doc_type=doc_type,
        )

        # 2. Branch A: Splice / Tamper Forensics
        tamper_result = self.tamper_detector.detect(
            image=original_image,
            doc_type=doc_type,
            forensic_context=forensic_context,
        )

        # 3. Branch B: Guilloché / Background Forensics
        guilloche_result = self.guilloche_detector.inspect_pattern(
            image=original_image,
            doc_type=doc_type,
            country=country,
            version=version,
            region=region,
            is_applicable=is_guilloche_applicable,
        )

        return tamper_result, guilloche_result, forensic_context
