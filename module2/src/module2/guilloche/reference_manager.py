"""Reference manager for Guilloché and authentic security background patterns.

Indexes and retrieves authentic reference patterns based on:
country, document type, version/series, and region.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from PIL import Image

from module2.preprocessing.image_loader import SafeImageLoader
from module2.utils.logger import get_logger

logger = get_logger("module2.guilloche.reference_manager")


class GuillocheReferenceManager:
    """Manages an indexed library of authentic reference security patterns."""

    def __init__(self, reference_root: Optional[Union[str, Path]] = None):
        """Initialize reference manager.

        Args:
            reference_root: Root directory containing authentic reference patterns.
        """
        self.reference_root = Path(reference_root) if reference_root else None
        self._in_memory_references: Dict[Tuple[str, str, str, str], Image.Image] = {}

    def register_reference(
        self,
        image: Image.Image,
        country: str,
        document_type: str,
        version: str = "default",
        region: str = "background",
    ) -> None:
        """Register an in-memory authentic reference pattern.

        Args:
            image: Clean PIL Image of the authentic security pattern.
            country: Country code or name (e.g. 'USA', 'IND', 'GBR').
            document_type: Document type name (e.g. 'passport', 'visa').
            version: Series or version identifier.
            region: Region identifier (e.g. 'background', 'guilloche_band').
        """
        key = self._make_key(country, document_type, version, region)
        self._in_memory_references[key] = image.copy()
        logger.debug(f"Registered reference for key: {key}")

    def get_reference(
        self,
        country: Optional[str],
        document_type: str,
        version: Optional[str] = "default",
        region: Optional[str] = "background",
    ) -> Optional[Image.Image]:
        """Retrieve authentic reference pattern for a document specification.

        Args:
            country: Country code or name.
            document_type: Document type name.
            version: Series or version identifier.
            region: Pattern region identifier.

        Returns:
            PIL.Image.Image copy if found, otherwise None.
        """
        if not country or not document_type:
            logger.debug(
                f"Reference lookup failed: country ({country}) or document_type ({document_type}) missing."
            )
            return None

        c = country.strip().upper()
        d = document_type.strip().lower()
        v = (version or "default").strip().lower()
        r = (region or "background").strip().lower()

        # Check in-memory references first
        key = (c, d, v, r)
        if key in self._in_memory_references:
            logger.debug(f"Found in-memory reference for {key}")
            return self._in_memory_references[key].copy()

        # Fallback to default version if specific version not found
        default_key = (c, d, "default", r)
        if default_key in self._in_memory_references:
            logger.debug(f"Found in-memory reference for default key {default_key}")
            return self._in_memory_references[default_key].copy()

        # Check filesystem library if reference_root is configured
        if self.reference_root and self.reference_root.exists():
            # Pattern search: {root}/{country}/{doc_type}_{version}_{region}.*
            candidate_names = [
                f"{d}_{v}_{r}.png",
                f"{d}_{v}_{r}.jpg",
                f"{d}_default_{r}.png",
                f"{d}_{r}.png",
                f"{d}.png",
            ]
            country_dir = self.reference_root / c
            if country_dir.exists() and country_dir.is_dir():
                for name in candidate_names:
                    file_path = country_dir / name
                    if file_path.exists() and file_path.is_file():
                        try:
                            ref_img = SafeImageLoader.load_image(file_path)
                            logger.debug(f"Loaded filesystem reference from {file_path}")
                            return ref_img
                        except Exception as e:
                            logger.warning(f"Error loading reference file {file_path}: {e}")

        logger.info(
            f"No authentic reference pattern found for country={c}, "
            f"doc_type={d}, version={v}, region={r}"
        )
        return None

    @staticmethod
    def _make_key(country: str, document_type: str, version: str, region: str) -> Tuple[str, str, str, str]:
        return (
            country.strip().upper(),
            document_type.strip().lower(),
            version.strip().lower(),
            region.strip().lower(),
        )
