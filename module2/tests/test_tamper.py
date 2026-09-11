"""Unit tests for SIDTD EfficientNet-B3 Tamper / Forgery Detector."""

import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from module2.models.efficientnet_detector import SIDTDEfficientNetDetector, build_efficientnet_b3
from module2.schemas.output_schema import TamperStatus
from module2.tamper.tamper_detector import TamperDetector


class TestTamperDetector(unittest.TestCase):
    """Test suite for tamper detection model loading, inference, and error handling."""

    @classmethod
    def setUpClass(cls):
        cls.test_image = Image.new("RGB", (300, 300), color=(128, 128, 128))
        cls.detector = TamperDetector(device="cpu")

    def test_model_architecture_and_forward(self):
        """Verify EfficientNet-B3 architecture instantiation and output dimensions."""
        model = build_efficientnet_b3(num_classes=2, pretrained=False)
        dummy_tensor = torch.randn(2, 3, 300, 300)
        output = model(dummy_tensor)
        self.assertEqual(output.shape, (2, 2))

    def test_checkpoint_loading(self):
        """Verify model checkpoint loader gracefully handles existing and missing files."""
        # Test with missing checkpoint path
        detector_missing = SIDTDEfficientNetDetector(checkpoint_path="non_existent.pth", device="cpu")
        self.assertIsNotNone(detector_missing.model)

        # Test with actual initialized checkpoint
        ckpt_path = Path(__file__).resolve().parent.parent / "models" / "sidtd_efficientnet_b3.pth"
        if ckpt_path.exists():
            detector_valid = SIDTDEfficientNetDetector(checkpoint_path=ckpt_path, device="cpu")
            self.assertIsNotNone(detector_valid.model)

    def test_tamper_inference_structured_output(self):
        """Verify structured output fields and data types."""
        result = self.detector.detect(self.test_image, doc_type="passport")

        self.assertIn("status", result)
        self.assertIn("label", result)
        self.assertIn("passed", result)
        self.assertIn("confidence", result)
        self.assertIn("forgery_probability", result)
        self.assertIn("bona_fide_probability", result)

        self.assertIn(result["status"], [TamperStatus.BONA_FIDE.value, TamperStatus.FORGED.value])
        self.assertIsInstance(result["passed"], bool)
        self.assertGreaterEqual(result["forgery_probability"], 0.0)
        self.assertLessEqual(result["forgery_probability"], 1.0)

    def test_tamper_inference_with_forensic_context(self):
        """Verify forensic signals are attached when provided."""
        forensic_context = {
            "ela_mean": 0.05,
            "ela_std": 0.02,
            "dct_features": {"ac_energy": 12.5, "high_freq_ratio": 0.15},
        }
        result = self.detector.detect(
            self.test_image, doc_type="visa", forensic_context=forensic_context
        )
        self.assertIn("forensic_signals", result)
        self.assertEqual(result["forensic_signals"]["ela_mean"], 0.05)

    def test_invalid_input_handling(self):
        """Verify error status is returned cleanly on corrupt/invalid input without crashing."""
        detector = TamperDetector(device="cpu")
        # Passing an invalid object as image
        result = detector.detect("not_an_image", doc_type="corrupt_doc")  # type: ignore
        self.assertEqual(result["status"], TamperStatus.ERROR.value)
        self.assertFalse(result["passed"])
        self.assertIn("error", result)

    def test_load_sidtd_split_from_csv(self):
        """Verify load_sidtd_split parses Split_normal CSVs and resolves template images."""
        import tempfile
        from training.train_sidtd import load_sidtd_split

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split_dir = root / "Split_normal"
            split_dir.mkdir(parents=True)

            images_dir = root / "templates" / "Images"
            (images_dir / "reals" / "ESP").mkdir(parents=True)
            (images_dir / "fakes" / "ESP").mkdir(parents=True)

            real_img = images_dir / "reals" / "ESP" / "real_1.jpg"
            fake_img = images_dir / "fakes" / "ESP" / "fake_1.jpg"
            real_img.write_bytes(b"dummy_jpeg_data")
            fake_img.write_bytes(b"dummy_jpeg_data")

            # Create train CSV
            train_csv = split_dir / "train_split_SIDTD.csv"
            train_csv.write_text(
                "label_name,label,image_path,class,class_name\n"
                "reals,0,templates/Images/reals/ESP/real_1.jpg,0,ESP_ID\n"
                "fakes,1,templates/Images/fakes/ESP/fake_1.jpg,0,ESP_ID\n"
            )

            # Create val CSV
            val_csv = split_dir / "val_split_SIDTD.csv"
            val_csv.write_text(
                "label_name,label,image_path,class,class_name\n"
                "reals,0,Images/reals/ESP/real_1.jpg,0,ESP_ID\n"
            )

            # Create test CSV
            test_csv = split_dir / "test_split_SIDTD.csv"
            test_csv.write_text(
                "label_name,label,image_path,class,class_name\n"
                "fakes,1,fakes/ESP/fake_1.jpg,0,ESP_ID\n"
            )

            train_samples, val_samples, test_samples = load_sidtd_split(root, "Split_normal")

            self.assertEqual(len(train_samples), 2)
            self.assertEqual(train_samples[0], (str(real_img.resolve()), 0))
            self.assertEqual(train_samples[1], (str(fake_img.resolve()), 1))

            self.assertEqual(len(val_samples), 1)
            self.assertEqual(val_samples[0], (str(real_img.resolve()), 0))

            self.assertEqual(len(test_samples), 1)
            self.assertEqual(test_samples[0], (str(fake_img.resolve()), 1))


if __name__ == "__main__":
    unittest.main()
