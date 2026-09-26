"""Raw document preprocessor for Module 2.

Handles real-world raw input images (mobile phone camera photos, desk captures, scans):
1. EXIF Auto-orientation (corrects orientation from smartphone cameras).
2. Document Boundary Detection & Perspective Rectification (removes desk/background).
3. Saliency / Foreground Document Cropping (fallback when 4 corners are obscured).
4. Illumination Normalization (CLAHE on L-channel in LAB space to remove room shadows).
5. Safe preservation of original image handle.
"""

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

from module2.utils.logger import get_logger

logger = get_logger("module2.preprocessing.raw_preprocessor")


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order 4 contour points in deterministic order: [top-left, top-right, bottom-right, bottom-left]."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has smallest sum
    rect[2] = pts[np.argmax(s)]  # bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right has smallest diff
    rect[3] = pts[np.argmax(diff)]  # bottom-left has largest diff
    return rect


class RawDocumentPreprocessor:
    """Preprocesses raw smartphone or scanner images into clean, rectified document images."""

    def __init__(
        self,
        enable_perspective_warp: bool = True,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        min_doc_area_ratio: float = 0.20,
    ):
        self.enable_perspective_warp = enable_perspective_warp
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.min_doc_area_ratio = min_doc_area_ratio

    def auto_orient(self, image: Image.Image) -> Image.Image:
        """Apply EXIF orientation tag if present (common in smartphone captures)."""
        try:
            return ImageOps.exif_transpose(image)
        except Exception:
            return image

    def detect_document_quadrilateral(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Attempt to find 4-corner document boundary polygon."""
        h, w = img_bgr.shape[:2]
        max_dim = max(h, w)
        scale = 800.0 / max_dim if max_dim > 800 else 1.0
        resized = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Strategy 1: Multi-threshold Canny edge detection
        v = np.median(blurred)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edged = cv2.Canny(blurred, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edged, kernel, iterations=2)

        total_area = resized.shape[0] * resized.shape[1]
        best_quad = None
        max_area = 0

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            area = cv2.contourArea(c)
            if len(approx) == 4 and area > (total_area * self.min_doc_area_ratio):
                if area > max_area:
                    max_area = area
                    best_quad = approx

        # Strategy 2: Adaptive Otsu thresholding if Canny didn't isolate 4 corners
        if best_quad is None:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            contours_otsu, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in sorted(contours_otsu, key=cv2.contourArea, reverse=True)[:5]:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.025 * peri, True)
                area = cv2.contourArea(c)
                if len(approx) == 4 and area > (total_area * self.min_doc_area_ratio):
                    if area > max_area:
                        max_area = area
                        best_quad = approx

        if best_quad is not None:
            # Scale back to original coordinates
            pts = best_quad.reshape(4, 2) / scale
            return pts
        return None

    def detect_salient_document_box(self, img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect rectangular document bounding box when perspective quad is not cleanly isolated."""
        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

        blurred = cv2.GaussianBlur(grad, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            total_area = h * w
            # If the detected document area is between 25% and 98% of the frame, crop it
            if 0.25 <= (area / total_area) <= 0.98:
                x, y, bw, bh = cv2.boundingRect(largest)
                # Add small 2% padding
                pad_x = int(bw * 0.02)
                pad_y = int(bh * 0.02)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + bw + pad_x)
                y2 = min(h, y + bh + pad_y)
                return (x1, y1, x2 - x1, y2 - y1)
        return None

    def warp_perspective(self, img_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Perform 4-point homography warp to obtain a flat rectangular document."""
        rect = order_quad_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        if max_width < 100 or max_height < 100:
            return img_bgr

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img_bgr, matrix, (max_width, max_height), flags=cv2.INTER_LANCZOS4)
        return warped

    def normalize_illumination(self, img_bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE in LAB color space to equalize room lighting shadows without color distortion."""
        try:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size,
            )
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception:
            return img_bgr

    def suppress_specular_glare(self, img_bgr: np.ndarray, threshold: int = 245) -> np.ndarray:
        """Detect and softly attenuate intense camera flash specular glare hotspots on passport laminates."""
        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Find specular glare mask (extreme brightness hotspots)
            _, glare_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            glare_pixel_count = np.count_nonzero(glare_mask)
            total_pixels = glare_mask.size

            # Only attenuate if glare covers a localized hotspot (<15% of the total document)
            if 0 < glare_pixel_count and (glare_pixel_count / total_pixels) < 0.15:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilated_mask = cv2.dilate(glare_mask, kernel, iterations=1)
                inpainted = cv2.inpaint(img_bgr, dilated_mask, 3, cv2.INPAINT_TELEA)
                # Soft blend: 75% inpainted attenuation, 25% original texture
                return cv2.addWeighted(inpainted, 0.75, img_bgr, 0.25, 0)
        except Exception:
            pass
        return img_bgr

    def preprocess_raw_document(
        self,
        raw_image: Image.Image,
        doc_type: str = "document",
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Main entry point: transform a raw mobile/scanner image into a clean document.

        Args:
            raw_image: Raw PIL Image as uploaded by user.
            doc_type: Document name for logging.

        Returns:
            Tuple of (rectified_clean_PIL_image, preprocessing_metadata).
        """
        metadata: Dict[str, Any] = {
            "doc_type": doc_type,
            "original_size": raw_image.size,
            "perspective_rectified": False,
            "boundary_cropped": False,
            "clahe_applied": False,
        }

        # 1. EXIF Auto-orientation
        oriented = self.auto_orient(raw_image).convert("RGB")

        # Convert to BGR for OpenCV processing
        img_np = np.array(oriented)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        cleaned_bgr = img_bgr

        # 2. Try 4-point quadrilateral detection and perspective warp
        if self.enable_perspective_warp:
            quad_pts = self.detect_document_quadrilateral(img_bgr)
            if quad_pts is not None:
                logger.info(f"[{doc_type}] Document 4-point quadrilateral detected -> Applying perspective warp.")
                cleaned_bgr = self.warp_perspective(img_bgr, quad_pts)
                metadata["perspective_rectified"] = True
            else:
                # Fallback to salient document bounding box
                box = self.detect_salient_document_box(img_bgr)
                if box is not None:
                    bx, by, bw, bh = box
                    logger.info(f"[{doc_type}] Document bounding box detected ({bw}x{bh}) -> Auto-cropping background.")
                    cleaned_bgr = img_bgr[by : by + bh, bx : bx + bw]
                    metadata["boundary_cropped"] = True
                    metadata["crop_box"] = box

        # 3. Specular glare attenuation (dampens camera flash hotspots on passport laminate)
        cleaned_bgr = self.suppress_specular_glare(cleaned_bgr)

        # 4. Optional Illumination normalization
        if self.enable_clahe:
            cleaned_bgr = self.normalize_illumination(cleaned_bgr)
            metadata["clahe_applied"] = True

        # Convert back to clean RGB PIL Image
        final_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)
        processed_pil = Image.fromarray(final_rgb)
        metadata["final_size"] = processed_pil.size

        return processed_pil, metadata
