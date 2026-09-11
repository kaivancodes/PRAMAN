"""Discrete Cosine Transform (DCT) frequency-domain representation for forensic analysis.

Computes 8x8 block-based 2D DCT and extracts frequency statistical features.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image


class DiscreteCosineTransform:
    """Computes 2D DCT and frequency energy statistics across image blocks."""

    def __init__(self, block_size: int = 8):
        """Initialize DCT extractor.

        Args:
            block_size: Size of 2D DCT blocks (standard JPEG uses 8x8).
        """
        self.block_size = block_size

    def compute_dct_map(self, image: Image.Image) -> np.ndarray:
        """Compute block-wise 2D DCT for grayscale/luminance channel.

        Args:
            image: PIL Image (will be converted to luminance without modifying original).

        Returns:
            2D numpy float32 array with block-wise DCT coefficients.
        """
        # Convert to grayscale luminance
        gray = np.array(image.convert("L"), dtype=np.float32)
        h, w = gray.shape

        # Pad to multiple of block_size if necessary
        b = self.block_size
        pad_h = (b - (h % b)) % b
        pad_w = (b - (w % b)) % b
        if pad_h > 0 or pad_w > 0:
            gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")

        h_pad, w_pad = gray.shape
        dct_map = np.zeros_like(gray, dtype=np.float32)

        for y in range(0, h_pad, b):
            for x in range(0, w_pad, b):
                block = gray[y : y + b, x : x + b]
                dct_block = cv2.dct(block)
                dct_map[y : y + b, x : x + b] = dct_block

        # Unpad back to original size if padded
        if pad_h > 0 or pad_w > 0:
            dct_map = dct_map[:h, :w]

        return dct_map

    def extract_frequency_features(self, image: Image.Image) -> Dict[str, float]:
        """Extract statistical frequency features from the image's DCT representation.

        Returns:
            Dictionary with:
              - dc_energy: Mean absolute energy in DC coefficients
              - ac_energy: Mean absolute energy in AC coefficients
              - high_freq_ratio: Ratio of high frequency to total AC energy
              - ac_variance: Variance of AC coefficients
        """
        gray = np.array(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        b = self.block_size

        pad_h = (b - (h % b)) % b
        pad_w = (b - (w % b)) % b
        if pad_h > 0 or pad_w > 0:
            gray = np.pad(gray, ((0, pad_h), (0, pad_w)), mode="reflect")

        h_pad, w_pad = gray.shape

        dc_coeffs = []
        ac_coeffs = []
        high_freq_coeffs = []

        for y in range(0, h_pad, b):
            for x in range(0, w_pad, b):
                block = gray[y : y + b, x : x + b]
                dct_block = cv2.dct(block)

                # DC is (0, 0)
                dc_coeffs.append(abs(dct_block[0, 0]))

                # AC is all non-(0,0)
                ac_mask = np.ones((b, b), dtype=bool)
                ac_mask[0, 0] = False
                ac_vals = np.abs(dct_block[ac_mask])
                ac_coeffs.extend(ac_vals)

                # High frequencies: lower triangle/bottom-right of 8x8 block (indices sum > 6)
                for r in range(b):
                    for c in range(b):
                        if r + c > 6:
                            high_freq_coeffs.append(abs(dct_block[r, c]))

        total_ac = float(np.sum(ac_coeffs)) if ac_coeffs else 1e-6
        high_ac = float(np.sum(high_freq_coeffs)) if high_freq_coeffs else 0.0

        return {
            "dc_energy": float(np.mean(dc_coeffs)) if dc_coeffs else 0.0,
            "ac_energy": float(np.mean(ac_coeffs)) if ac_coeffs else 0.0,
            "ac_variance": float(np.var(ac_coeffs)) if ac_coeffs else 0.0,
            "high_freq_ratio": float(high_ac / max(1e-6, total_ac)),
        }
