"""Image loader and validator for Module 2.

Handles decoding, EXIF orientation, RGB conversion, format validation,
and safe handling of corrupt/missing files while preserving original resolution.
"""

import io
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageOps

from module2.utils.logger import get_logger

logger = get_logger("module2.preprocessing.image_loader")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


class ImageLoadError(Exception):
    """Base exception for image loading failures."""
    pass


class CorruptImageError(ImageLoadError):
    """Raised when an image file is corrupt or truncated."""
    pass


class UnsupportedFormatError(ImageLoadError):
    """Raised when the image format is not supported."""
    pass


class SafeImageLoader:
    """Safely loads and validates document images preserving original resolution."""

    @staticmethod
    def load_image(
        image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
        to_rgb: bool = True,
    ) -> Image.Image:
        """Load and return an image as a fresh PIL Image in RGB mode.

        Args:
            image_input: File path, Path object, PIL Image, numpy array, or raw bytes.
            to_rgb: If True, converts image to RGB mode.

        Returns:
            PIL.Image.Image instance preserving original resolution.

        Raises:
            ImageLoadError: If image cannot be loaded or is invalid.
            CorruptImageError: If image data is corrupted.
            UnsupportedFormatError: If file extension is unsupported.
        """
        if image_input is None:
            raise ImageLoadError("Image input is None.")

        # Handle PIL Image
        if isinstance(image_input, Image.Image):
            try:
                img = image_input.copy()
                img = ImageOps.exif_transpose(img)
                if to_rgb and img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except Exception as e:
                raise CorruptImageError(f"Failed to process provided PIL Image: {str(e)}") from e

        # Handle numpy ndarray
        if isinstance(image_input, np.ndarray):
            try:
                if image_input.size == 0:
                    raise CorruptImageError("Empty numpy array provided as image input.")
                # If grayscale or 2D
                if image_input.ndim == 2:
                    img = Image.fromarray(image_input)
                elif image_input.ndim == 3:
                    # Check channel count
                    if image_input.shape[2] == 3:
                        img = Image.fromarray(image_input)
                    elif image_input.shape[2] == 4:
                        img = Image.fromarray(image_input).convert("RGB")
                    elif image_input.shape[2] == 1:
                        img = Image.fromarray(image_input.squeeze(-1))
                    else:
                        raise UnsupportedFormatError(f"Unsupported channel dimension: {image_input.shape}")
                else:
                    raise UnsupportedFormatError(f"Unsupported array dimensions: {image_input.ndim}")

                if to_rgb and img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except Exception as e:
                raise CorruptImageError(f"Failed to convert numpy array to image: {str(e)}") from e

        # Handle raw bytes
        if isinstance(image_input, bytes):
            if len(image_input) == 0:
                raise CorruptImageError("Provided image bytes are empty.")
            try:
                stream = io.BytesIO(image_input)
                img = Image.open(stream)
                img.verify()  # Validate image integrity
                stream.seek(0)
                img = Image.open(stream)
                img.load()  # Fully decode into memory
                img = ImageOps.exif_transpose(img)
                if to_rgb and img.mode != "RGB":
                    img = img.convert("RGB")
                return img
            except Exception as e:
                raise CorruptImageError(f"Corrupt or invalid image bytes: {str(e)}") from e

        # Handle file paths
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise ImageLoadError(f"Image file does not exist: {path}")
            if not path.is_file():
                raise ImageLoadError(f"Path is not a valid file: {path}")

            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise UnsupportedFormatError(
                    f"Unsupported image extension '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
                )

            try:
                with open(path, "rb") as f:
                    data = f.read()
                return SafeImageLoader.load_image(data, to_rgb=to_rgb)
            except ImageLoadError:
                raise
            except Exception as e:
                raise CorruptImageError(f"Failed to read image from path '{path}': {str(e)}") from e

        raise UnsupportedFormatError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def to_numpy_rgb(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to RGB numpy array without modifying original."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image, dtype=np.uint8)
