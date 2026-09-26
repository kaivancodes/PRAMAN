"""Unit tests for Siamese Guilloché, Tamper Training components, and AI Gate bypass."""

import unittest
import torch
from PIL import Image

from module2.guilloche.guilloche_detector import GuillocheDetector
from module2.guilloche.reference_manager import GuillocheReferenceManager
from module2.guilloche.similarity import GuillocheSimilarityEvaluator
from module2.models.siamese_guilloche import GuillocheSiameseDetector, SiameseResNet18
from module2.models.efficientnet_detector import SIDTDEfficientNetDetector
from module2.orchestrator.module2_orchestrator import Module2Orchestrator
from module2.schemas.input_schema import CaseInput
from module2.schemas.output_schema import AIGateStatus, CaseStatus, GuillocheStatus
from module2.scoring.forensic_scorer import ForensicScorer
from module2.document.document_processor import DocumentProcessor, ModularAIDetector
from module2.tamper.tamper_detector import TamperDetector


class TestModelsAndTraining(unittest.TestCase):
    """Test suite for new model architectures, evaluators, and AI gate bypass."""

    def setUp(self):
        self.device = torch.device("cpu")

    def test_siamese_resnet18_architecture(self):
        """Verify Siamese ResNet-18 forward pass and L2 normalization."""
        model = SiameseResNet18(embedding_dim=128, pretrained=False)
        model.eval()

        dummy_crop1 = torch.randn(2, 3, 128, 128)
        dummy_crop2 = torch.randn(2, 3, 128, 128)

        # Single image embedding
        emb = model(dummy_crop1)
        self.assertEqual(emb.shape, (2, 128))

        # Check L2 normalization: norm of each vector must be 1.0
        norms = torch.norm(emb, p=2, dim=-1).detach()
        for n in norms:
            self.assertAlmostEqual(float(n), 1.0, places=4)

        # Pairwise embeddings
        emb1, emb2 = model(dummy_crop1, dummy_crop2)
        self.assertEqual(emb1.shape, (2, 128))
        self.assertEqual(emb2.shape, (2, 128))

    def test_guilloche_similarity_evaluator_with_siamese(self):
        """Verify GuillocheSimilarityEvaluator correctly uses Siamese ResNet-18."""
        evaluator = GuillocheSimilarityEvaluator(
            embedding_dim=128,
            model_name="resnet18",
            device=self.device,
        )

        img1 = Image.new("RGB", (128, 128), color=(200, 200, 200))
        img2 = Image.new("RGB", (128, 128), color=(200, 200, 200))

        sim = evaluator.compute_similarity(img1, img2)
        self.assertIsInstance(sim, float)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)
        # Identical images must yield a similarity close to 1.0
        self.assertGreater(sim, 0.90)

    def test_ai_gate_bypass_in_orchestrator(self):
        """Verify that when ai_gate_enabled is False, AI check is automatically PASSED."""
        doc_proc = DocumentProcessor(
            ai_detector=ModularAIDetector(),
            tamper_detector=TamperDetector(device=self.device),
            guilloche_detector=GuillocheDetector(),
            device=self.device,
        )

        orchestrator = Module2Orchestrator(
            document_processor=doc_proc,
            scorer=ForensicScorer(),
            max_workers=1,
            ai_gate_enabled=False,  # AI part is done / bypassed
        )

        test_img = Image.new("RGB", (200, 200), color=(180, 180, 180))
        case_input = CaseInput(
            uuid="case-test-bypass",
            documents_present=["passport", "visa", "national_id"],
            passport_image=test_img,
            visa_image=test_img,
            national_id_image=test_img,
        )

        output = orchestrator.process_case(case_input)

        # Case must NOT be red-flagged by AI gate
        self.assertEqual(output.ai_generation_check, AIGateStatus.PASSED.value)
        self.assertEqual(output.status, CaseStatus.COMPLETED.value)
        self.assertIsNotNone(output.forensic_pass_score)
        self.assertIn("passport", output.document_results)
        self.assertIn("visa", output.document_results)
        self.assertIn("national_id", output.document_results)

    def test_ai_gate_simulation_still_flags_when_bypassed(self):
        """Verify that explicit testing flags (e.g. for red flag tests) are still honored."""
        doc_proc = DocumentProcessor(
            ai_detector=ModularAIDetector(),
            tamper_detector=TamperDetector(device=self.device),
            guilloche_detector=GuillocheDetector(),
            device=self.device,
        )
        # Explicitly flag passport for testing
        doc_proc.ai_detector.set_flagged_for_testing("passport")

        orchestrator = Module2Orchestrator(
            document_processor=doc_proc,
            scorer=ForensicScorer(),
            max_workers=1,
            ai_gate_enabled=False,
        )

        test_img = Image.new("RGB", (200, 200), color=(180, 180, 180))
        case_input = CaseInput(
            uuid="case-test-sim-flag",
            documents_present=["passport", "visa", "national_id"],
            passport_image=test_img,
            visa_image=test_img,
            national_id_image=test_img,
        )

        output = orchestrator.process_case(case_input)

        # Simulation must raise RED FLAG immediately
        self.assertEqual(output.status, CaseStatus.RED_FLAG.value)
        self.assertEqual(output.flagged_document, "passport")
        self.assertIsNone(output.forensic_pass_score)


if __name__ == "__main__":
    unittest.main()
