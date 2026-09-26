"""Tamper / Manipulation Forensics production wrapper.

Production input has already passed common document normalization (boundary detection/cropping, perspective correction, and CLAHE). The production upload-quality gate is separate. This wrapper performs only Tamper-specific preprocessing:
RGB + ELA + DCT -> 5-channel tensor -> EfficientNet-B3.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
from PIL import Image

from module2.preprocessing.ela import ErrorLevelAnalysis
from module2.preprocessing.dct import DiscreteCosineTransform
from module2.models.efficientnet_detector import SIDTDEfficientNetDetector
from module2.schemas.output_schema import TamperStatus
from module2.utils.logger import get_logger

logger = get_logger("module2.tamper.tamper_detector")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TamperDetector:
    def __init__(
        self,
        detector: Optional[SIDTDEfficientNetDetector] = None,
        checkpoint_path: Optional[str] = None,
        input_size: int = 300,
        tamper_threshold: float = 0.50,
        reason_threshold: float = 0.50,
        device: Optional[str] = "auto",
    ):
        self.detector = detector or SIDTDEfficientNetDetector(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self.input_size = input_size
        self.tamper_threshold = tamper_threshold
        self.reason_threshold = reason_threshold
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        img = image.convert("RGB").resize(
            (self.input_size, self.input_size),
            Image.Resampling.BILINEAR,
        )

        rgb = torch.from_numpy(
            np.asarray(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)
        rgb = (rgb - self.mean) / self.std

        ela = ErrorLevelAnalysis(
            default_quality=90,
            default_scale=10,
        ).compute_ela(img, quality=90, scale=10).convert("L")
        ela_t = torch.from_numpy(
            np.asarray(ela, dtype=np.float32) / 255.0
        ).unsqueeze(0)

        dct = DiscreteCosineTransform(block_size=8).compute_dct_map(img)
        dct = np.log1p(np.abs(dct))
        dct = (dct - dct.min()) / max(float(dct.max() - dct.min()), 1e-6)
        dct_t = torch.from_numpy(dct.astype(np.float32)).unsqueeze(0)

        return torch.cat([rgb, ela_t, dct_t], dim=0).unsqueeze(0)

    def detect(
        self,
        image: Image.Image,
        doc_type: str = "document",
        forensic_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        try:
            tensor = self._preprocess(image)
            output = self.detector.forward(tensor)

            forged_prob = float(output["forged_probability"])
            bona_fide_prob = float(output["bona_fide_probability"])
            reason_probs = output["reason_probabilities"]

            reasons = [
                name
                for name, probability in reason_probs.items()
                if probability >= self.reason_threshold
            ]

            is_forged = forged_prob >= self.tamper_threshold
            status = TamperStatus.FORGED.value if is_forged else TamperStatus.BONA_FIDE.value

            # Reasons are only failure reasons when the binary Tamper decision fails.
            failure_reasons = reasons if is_forged else []

            return {
                "status": status,
                "label": "FORGED" if is_forged else "BONA_FIDE",
                "passed": not is_forged,
                "confidence": round(forged_prob if is_forged else bona_fide_prob, 4),
                "forgery_probability": round(forged_prob, 4),
                "bona_fide_probability": round(bona_fide_prob, 4),
                "manipulation_reasons": {
                    name: round(float(probability), 4)
                    for name, probability in reason_probs.items()
                },
                "failure_reasons": failure_reasons,
                "reason_threshold": self.reason_threshold,
            }

        except Exception as e:
            logger.error("Tamper inference failed for %s: %s", doc_type, e)
            return {
                "status": TamperStatus.ERROR.value,
                "label": "ERROR",
                "passed": False,
                "failure_reasons": [],
                "error": str(e),
            }
