"""Tamper / Forgery detector wrapper using SIDTD EfficientNet-B3 model.

Processes document images and returns structured tamper results.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from PIL import Image

from module2.models.efficientnet_detector import SIDTDEfficientNetDetector
from module2.schemas.output_schema import TamperStatus
from module2.utils.logger import get_logger

logger = get_logger("module2.tamper.tamper_detector")

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TamperDetector:
    """Inference wrapper for SIDTD document forgery detection.

    Keeps the detector loaded in memory across documents.
    """

    def __init__(
        self,
        detector: Optional[SIDTDEfficientNetDetector] = None,
        checkpoint_path: Optional[str] = None,
        input_size: int = 300,
        tamper_threshold: float = 0.50,
        device: Optional[str] = "auto",
    ):
        """Initialize TamperDetector.

        Args:
            detector: Pre-instantiated SIDTDEfficientNetDetector.
            checkpoint_path: Path to checkpoint if detector is not provided.
            input_size: Dimension to resize image for EfficientNet-B3 (default 300).
            tamper_threshold: Probability threshold above which document is considered FORGED.
            device: Compute device identifier.
        """
        if detector is not None:
            self.detector = detector
        else:
            self.detector = SIDTDEfficientNetDetector(
                checkpoint_path=checkpoint_path,
                device=device,
            )

        self.input_size = input_size
        self.tamper_threshold = tamper_threshold
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL Image for EfficientNet-B3 inference without modifying original."""
        img_resized = image.convert("RGB").resize(
            (self.input_size, self.input_size), Image.Resampling.BILINEAR
        )
        arr = np.array(img_resized, dtype=np.float32) / 255.0  # H, W, C
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # 3, H, W
        tensor = (tensor - self.mean) / self.std
        return tensor.unsqueeze(0)  # 1, 3, H, W

    def detect(
        self,
        image: Image.Image,
        doc_type: str = "document",
        forensic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform tamper / forgery inference on the document image.

        Args:
            image: Original PIL Image.
            doc_type: Document type name.
            forensic_context: Optional forensic package (e.g. ELA/DCT statistics).

        Returns:
            Structured tamper dictionary.
        """
        try:
            logger.debug(f"Running tamper detector on {doc_type}")
            input_tensor = self._preprocess(image)
            probs = self.detector.forward_probs(input_tensor)
            probs_np = probs.cpu().numpy()[0]

            bona_fide_prob = float(probs_np[0])
            forged_prob = float(probs_np[1])

            is_forged = forged_prob >= self.tamper_threshold
            status = TamperStatus.FORGED.value if is_forged else TamperStatus.BONA_FIDE.value
            passed = not is_forged
            confidence = forged_prob if is_forged else bona_fide_prob

            result = {
                "status": status,
                "label": "FORGED" if is_forged else "BONA_FIDE",
                "passed": passed,
                "confidence": round(confidence, 4),
                "forgery_probability": round(forged_prob, 4),
                "bona_fide_probability": round(bona_fide_prob, 4),
            }

            if forensic_context:
                result["forensic_signals"] = {
                    "ela_mean": forensic_context.get("ela_mean"),
                    "ela_std": forensic_context.get("ela_std"),
                    "dct_features": forensic_context.get("dct_features"),
                }

            logger.info(
                f"Tamper check for {doc_type}: {status} "
                f"(forgery_prob: {forged_prob:.4f}, passed: {passed})"
            )
            return result

        except Exception as e:
            logger.error(f"Error during tamper detection for {doc_type}: {e}")
            return {
                "status": TamperStatus.ERROR.value,
                "label": "ERROR",
                "passed": False,
                "confidence": 0.0,
                "forgery_probability": 1.0,
                "bona_fide_probability": 0.0,
                "error": str(e),
            }
