"""Pattern extractor for Guilloché and security background features.

Extracts background ROIs and pattern regions while preserving original document images.
"""

from typing import Optional, Tuple
from PIL import Image

from module2.utils.logger import get_logger

logger = get_logger("module2.guilloche.pattern_extractor")


class PatternExtractor:
    """Extracts background security patterns from document images."""

    def __init__(self, crop_size: Tuple[int, int] = (256, 256)):
        """Initialize PatternExtractor.

        Args:
            crop_size: Target (width, height) for extracted pattern crops.
        """
        self.crop_size = crop_size

    def extract_pattern(
        self,
        image: Image.Image,
        roi_box: Optional[Tuple[int, int, int, int]] = None,
        region_type: str = "background",
    ) -> Image.Image:
        """Extract a security pattern region from the document image.

        Args:
            image: Original PIL Image.
            roi_box: Optional explicit (left, top, right, bottom) bounding box.
            region_type: Logical region identifier ('background', 'center', etc.).

        Returns:
            Extracted and standardized PIL Image pattern crop.
        """
        w, h = image.size

        if roi_box is not None:
            left, top, right, bottom = roi_box
            # Clamp coordinates safely
            left = max(0, min(left, w - 1))
            top = max(0, min(top, h - 1))
            right = max(left + 1, min(right, w))
            bottom = max(top + 1, min(bottom, h))
            crop = image.crop((left, top, right, bottom))
        else:
            # Default background crop: central area avoiding typical photo (left) and MRZ (bottom)
            # Typically centered at 30% to 70% width, 25% to 65% height
            left = int(w * 0.35)
            top = int(h * 0.25)
            right = int(w * 0.75)
            bottom = int(h * 0.65)
            crop = image.crop((left, top, right, bottom))

        # Standardize crop size for similarity encoding
        pattern_crop = crop.resize(self.crop_size, Image.Resampling.BILINEAR)
        logger.debug(f"Extracted pattern crop of size {pattern_crop.size} for region '{region_type}'")
        return pattern_crop
