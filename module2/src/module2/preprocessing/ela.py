"""Error Level Analysis (ELA) implementation for forensic inspection.

Identifies compression error discrepancies across image regions.
Note: ELA provides a forensic signal and should be evaluated alongside other features.
"""

import io
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


class ErrorLevelAnalysis:
    """Computes Error Level Analysis (ELA) representations from document images."""

    def __init__(self, default_quality: int = 90, default_scale: int = 10):
        """Initialize ELA processor.

        Args:
            default_quality: JPEG compression quality for recompression (1-100).
            default_scale: Multiplier to amplify pixel differences for visibility/features.
        """
        self.quality = default_quality
        self.scale = default_scale

    def compute_ela(
        self,
        image: Image.Image,
        quality: Optional[int] = None,
        scale: Optional[int] = None,
    ) -> Image.Image:
        """Generate ELA representation of an image without mutating the original.

        Args:
            image: Original PIL Image.
            quality: Optional override for compression quality.
            scale: Optional override for amplification factor.

        Returns:
            Amplified ELA image as a PIL Image.
        """
        q = quality if quality is not None else self.quality
        s = scale if scale is not None else self.scale

        # Ensure working on RGB copy
        orig = image.convert("RGB") if image.mode != "RGB" else image.copy()

        # Re-save to controlled in-memory JPEG
        buffer = io.BytesIO()
        orig.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        recompressed = Image.open(buffer)
        recompressed.load()

        # Calculate pixel difference
        diff = ImageChops.difference(orig, recompressed)

        # Calculate extreme differences to scale safely
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema]) if extrema else 1
        scale_factor = s if s > 0 else (255.0 / max(1, max_diff))

        # Amplify difference
        diff = ImageEnhance.Brightness(diff).enhance(scale_factor)
        return diff

    def compute_ela_features(
        self,
        image: Image.Image,
        quality: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, float]:
        """Compute ELA array and basic statistical signals.

        Returns:
            Tuple of:
              - ela_array: normalized float array [0, 1]
              - mean_error: float mean error level
              - std_error: float standard deviation of error
        """
        ela_img = self.compute_ela(image, quality=quality, scale=1)
        arr = np.array(ela_img, dtype=np.float32) / 255.0
        mean_err = float(np.mean(arr))
        std_err = float(np.std(arr))
        return arr, mean_err, std_err
