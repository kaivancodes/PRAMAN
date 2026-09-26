"""Forensic preprocessor coordinating multi-domain representations.

Extracts ELA, DCT, and document crops while preserving original image objects.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from module2.preprocessing.dct import DiscreteCosineTransform
from module2.preprocessing.ela import ErrorLevelAnalysis
from module2.utils.logger import get_logger

logger = get_logger("module2.preprocessing.forensic_preprocessor")


class ForensicPreprocessor:
    """Prepares forensic representations without mutating the original document image."""

    def __init__(
        self,
        ela_quality: int = 90,
        ela_scale: int = 10,
        dct_block_size: int = 8,
    ):
        self.ela = ErrorLevelAnalysis(default_quality=ela_quality, default_scale=ela_scale)
        self.dct = DiscreteCosineTransform(block_size=dct_block_size)

    def extract_forensic_package(
        self,
        original_image: Image.Image,
        doc_type: str,
    ) -> Dict[str, Any]:
        """Extract multi-modal forensic representations.

        Args:
            original_image: The clean, decoded original PIL Image.
            doc_type: Document type name for context.

        Returns:
            Dictionary containing:
              - original: clean PIL Image
              - ela_image: PIL Image of amplified ELA
              - ela_features: statistical ELA features (mean, std)
              - dct_features: frequency energy features (dc_energy, ac_energy, high_freq_ratio)
              - resolution: (width, height)
        """
        # Ensure working on a separate handle to guarantee original preservation
        safe_copy = original_image.copy()

        width, height = safe_copy.size
        logger.debug(f"Extracting forensic features for {doc_type} ({width}x{height})")

        # ELA representation
        ela_img = self.ela.compute_ela(safe_copy)
        _, ela_mean, ela_std = self.ela.compute_ela_features(safe_copy)

        # DCT features
        dct_features = self.dct.extract_frequency_features(safe_copy)

        return {
            "original": safe_copy,
            "ela_image": ela_img,
            "ela_mean": ela_mean,
            "ela_std": ela_std,
            "dct_features": dct_features,
            "resolution": (width, height),
        }

    @staticmethod
    def crop_region(
        image: Image.Image,
        box: Tuple[int, int, int, int],
    ) -> Image.Image:
        """Extract an ROI crop while preserving the original image.

        Args:
            image: PIL Image.
            box: (left, top, right, bottom) coordinates.

        Returns:
            Cropped PIL Image copy.
        """
        return image.crop(box)
