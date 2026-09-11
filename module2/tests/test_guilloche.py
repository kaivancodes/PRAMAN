"""Unit tests for Guilloché / Background pattern forensics."""

import unittest
from PIL import Image

from module2.guilloche.guilloche_detector import GuillocheDetector
from module2.guilloche.pattern_extractor import PatternExtractor
from module2.guilloche.reference_manager import GuillocheReferenceManager
from module2.guilloche.similarity import GuillocheSimilarityEvaluator
from module2.schemas.output_schema import GuillocheStatus


class TestGuillocheForensics(unittest.TestCase):
    """Test suite for reference manager, pattern extractor, and guilloche detector."""

    def setUp(self):
        # Create patterned images for testing
        self.ref_pattern = Image.new("RGB", (256, 256), color=(200, 220, 240))
        # Draw some variation into reference
        for x in range(0, 256, 16):
            for y in range(0, 256, 16):
                self.ref_pattern.putpixel((x, y), (50, 100, 150))

        self.query_doc = Image.new("RGB", (600, 400), color=(200, 220, 240))
        for x in range(0, 600, 16):
            for y in range(0, 400, 16):
                self.query_doc.putpixel((x, y), (50, 100, 150))

        self.ref_manager = GuillocheReferenceManager()
        self.ref_manager.register_reference(
            image=self.ref_pattern,
            country="USA",
            document_type="passport",
            version="default",
            region="background",
        )

        self.detector = GuillocheDetector(
            reference_manager=self.ref_manager,
            similarity_threshold=0.70,
        )

    def test_reference_found_consistent(self):
        """When matching reference exists and pattern is identical/similar, status is CONSISTENT."""
        result = self.detector.inspect_pattern(
            image=self.query_doc,
            doc_type="passport",
            country="USA",
            version="default",
            region="background",
        )
        self.assertEqual(result["status"], GuillocheStatus.CONSISTENT.value)
        self.assertTrue(result["passed"])
        self.assertTrue(result["reference_available"])
        self.assertIsNotNone(result["similarity_score"])
        self.assertGreaterEqual(result["similarity_score"], 0.70)

    def test_reference_missing_returns_reference_required(self):
        """When no reference exists for document, return REFERENCE_REQUIRED, never fake/fail."""
        result = self.detector.inspect_pattern(
            image=self.query_doc,
            doc_type="passport",
            country="FRA",  # No France reference registered
            version="default",
            region="background",
        )
        self.assertEqual(result["status"], GuillocheStatus.REFERENCE_REQUIRED.value)
        self.assertIsNone(result["passed"])
        self.assertFalse(result["reference_available"])
        self.assertIsNone(result["similarity_score"])

    def test_not_applicable_document(self):
        """When document feature is not applicable, return NOT_APPLICABLE status."""
        result = self.detector.inspect_pattern(
            image=self.query_doc,
            doc_type="permit",
            country="USA",
            is_applicable=False,
        )
        self.assertEqual(result["status"], GuillocheStatus.NOT_APPLICABLE.value)
        self.assertIsNone(result["passed"])
        self.assertFalse(result["reference_available"])

    def test_inconsistent_pattern(self):
        """When pattern differs substantially from reference, status is INCONSISTENT."""
        # Create a completely inverted / black query document
        black_query = Image.new("RGB", (600, 400), color=(0, 0, 0))

        # Use strict threshold to guarantee inconsistency on completely different pattern
        detector_strict = GuillocheDetector(
            reference_manager=self.ref_manager,
            similarity_threshold=0.9999,  # Very high threshold
        )
        result = detector_strict.inspect_pattern(
            image=black_query,
            doc_type="passport",
            country="USA",
        )
        self.assertEqual(result["status"], GuillocheStatus.INCONSISTENT.value)
        self.assertFalse(result["passed"])
        self.assertTrue(result["reference_available"])
        self.assertIsNotNone(result["similarity_score"])

    def test_pattern_extractor_preserves_original(self):
        """Pattern extraction should not mutate original image."""
        orig_copy = self.query_doc.copy()
        extractor = PatternExtractor(crop_size=(128, 128))
        crop = extractor.extract_pattern(self.query_doc)

        self.assertEqual(crop.size, (128, 128))
        self.assertEqual(self.query_doc.size, orig_copy.size)
        self.assertEqual(list(self.query_doc.getdata()), list(orig_copy.getdata()))


if __name__ == "__main__":
    unittest.main()
