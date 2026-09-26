"""Integration and case-level tests for Module 2 Orchestrator.

Tests all mandated cases:
Case 1: Passport NO, Visa NO, National ID NO -> Module 2.2 executes, final score generated
Case 2: Passport YES -> RED FLAG, STOP, no Module 2.2, no final score
Case 3: Passport NO, Visa YES -> RED FLAG, STOP, national_id not processed, no final score
Case 4: Passport NO, Visa NO, National ID YES -> RED FLAG, STOP, optional docs not processed, no final score
Case 5: Passport NO, Visa NO, National ID NO, Driving License YES -> RED FLAG, STOP, no final score
Case 6: Passport NO, Visa NO, National ID NO, Driving License NO, Permit NO -> Module 2.2 completes, final score generated
"""

import unittest
from PIL import Image

from module2.document.document_processor import DocumentProcessor, ModularAIDetector
from module2.orchestrator.module2_orchestrator import Module2Orchestrator
from module2.schemas.input_schema import CaseInput
from module2.schemas.output_schema import AIGateStatus, CaseStatus


class TestModule2Workflow(unittest.TestCase):
    """End-to-end integration tests for Module 2 case workflows and early stop."""

    def setUp(self):
        self.img = Image.new("RGB", (200, 200), color=(180, 180, 180))
        self.ai_detector = ModularAIDetector()
        self.doc_processor = DocumentProcessor(ai_detector=self.ai_detector)
        self.orchestrator = Module2Orchestrator(
            document_processor=self.doc_processor,
            max_workers=1,
        )

    def test_case_1_all_compulsory_pass_ai_gate(self):
        """Case 1: Passport NO, Visa NO, National ID NO -> Module 2.2 executes, final score generated."""
        self.ai_detector.clear_flags_for_testing()

        case_input = CaseInput(
            uuid="case-001-all-pass",
            documents_present=["passport", "visa", "national_id"],
            passport_image=self.img,
            visa_image=self.img,
            national_id_image=self.img,
        )

        output = self.orchestrator.process_case(case_input)

        self.assertEqual(output.status, CaseStatus.COMPLETED.value)
        self.assertEqual(output.ai_generation_check, AIGateStatus.PASSED.value)
        self.assertIsNotNone(output.forensic_pass_score)
        self.assertGreaterEqual(output.forensic_pass_score, 0.0)
        self.assertLessEqual(output.forensic_pass_score, 100.0)
        self.assertIsNone(output.flagged_document)
        self.assertEqual(output.documents_analyzed, ["passport", "visa", "national_id"])
        self.assertIn("passport", output.document_results)
        self.assertIn("visa", output.document_results)
        self.assertIn("national_id", output.document_results)
        self.assertIn("tamper_result", output.document_results["passport"])

    def test_case_2_passport_ai_generated_stops_immediately(self):
        """Case 2: Passport YES -> RED FLAG, STOP, no Module 2.2, no final score."""
        self.ai_detector.clear_flags_for_testing()
        self.ai_detector.set_flagged_for_testing("passport")

        case_input = CaseInput(
            uuid="case-002-passport-ai",
            documents_present=["passport", "visa", "national_id"],
            passport_image=self.img,
            visa_image=self.img,
            national_id_image=self.img,
        )

        output = self.orchestrator.process_case(case_input)

        self.assertEqual(output.status, CaseStatus.RED_FLAG.value)
        self.assertEqual(output.ai_generation_check, AIGateStatus.FAILED.value)
        self.assertEqual(output.flag_type, "AI_GENERATED_DOCUMENT")
        self.assertEqual(output.flagged_document, "passport")
        self.assertIsNone(output.forensic_pass_score)

        # Ensure visa and national_id were NOT processed
        self.assertNotIn("visa", output.document_results)
        self.assertNotIn("national_id", output.document_results)

    def test_case_3_visa_ai_generated_stops_at_visa(self):
        """Case 3: Passport NO, Visa YES -> RED FLAG, STOP, national_id not processed, no final score."""
        self.ai_detector.clear_flags_for_testing()
        self.ai_detector.set_flagged_for_testing("visa")

        case_input = CaseInput(
            uuid="case-003-visa-ai",
            documents_present=["passport", "visa", "national_id"],
            passport_image=self.img,
            visa_image=self.img,
            national_id_image=self.img,
        )

        output = self.orchestrator.process_case(case_input)

        self.assertEqual(output.status, CaseStatus.RED_FLAG.value)
        self.assertEqual(output.ai_generation_check, AIGateStatus.FAILED.value)
        self.assertEqual(output.flagged_document, "visa")
        self.assertIsNone(output.forensic_pass_score)

        # Passport was checked and passed
        self.assertIn("passport", output.document_results)
        self.assertEqual(output.document_results["passport"]["ai_gate_status"], AIGateStatus.PASSED.value)
        # Visa was flagged
        self.assertIn("visa", output.document_results)
        self.assertEqual(output.document_results["visa"]["ai_gate_status"], AIGateStatus.FAILED.value)
        # National ID must NOT have been processed
        self.assertNotIn("national_id", output.document_results)

    def test_case_4_national_id_ai_generated_stops_before_optional(self):
        """Case 4: Passport NO, Visa NO, National ID YES -> RED FLAG, STOP, optional docs not processed, no final score."""
        self.ai_detector.clear_flags_for_testing()
        self.ai_detector.set_flagged_for_testing("national_id")

        case_input = CaseInput(
            uuid="case-004-id-ai",
            documents_present=["passport", "visa", "national_id", "driving_license", "permit"],
            passport_image=self.img,
            visa_image=self.img,
            national_id_image=self.img,
            driving_license_image=self.img,
            permit_image=self.img,
        )

        output = self.orchestrator.process_case(case_input)

        self.assertEqual(output.status, CaseStatus.RED_FLAG.value)
        self.assertEqual(output.flagged_document, "national_id")
        self.assertIsNone(output.forensic_pass_score)

        # Optional documents must NOT be processed
        self.assertNotIn("driving_license", output.document_results)
        self.assertNotIn("permit", output.document_results)

    def test_case_5_optional_driving_license_ai_generated_red_flag(self):
        """Case 5: Passport NO, Visa NO, National ID NO, Driving License YES -> RED FLAG, STOP, no final score."""
        self.ai_detector.clear_flags_for_testing()
        self.ai_detector.set_flagged_for_testing("driving_license")

        case_input = CaseInput(
            uuid="case-005-dl-ai",
            documents_present=["passport", "visa", "national_id", "driving_license", "permit"],
            passport_image=self.img,
            visa_image=self.img,
            national_id_image=self.img,
            driving_license_image=self.img,
            permit_image=self.img,
        )

        output = self.orchestrator.process_case(case_input)

        self.assertEqual(output.status, CaseStatus.RED_FLAG.value)
        self.assertEqual(output.flagged_document, "driving_license")
        self.assertIsNone(output.forensic_pass_score)

        # Compulsory docs passed
        self.assertEqual(output.document_results["passport"]["ai_gate_status"], AIGateStatus.PASSED.value)
        self.assertEqual(output.document_results["visa"]["ai_gate_status"], AIGateStatus.PASSED.value)
        self.assertEqual(output.document_results["national_id"]["ai_gate_status"], AIGateStatus.PASSED.value)
        # Permit must NOT be processed
        self.assertNotIn("permit", output.document_results)

    def test_case_6_all_5_documents_pass_generates_score(self):
        """Case 6: Passport NO, Visa NO, National ID NO, Driving License NO, Permit NO -> Module 2.2 completes, final score generated."""
        self.ai_detector.clear_flags_for_testing()

        case_input = CaseInput(
            uuid="case-006-all-5-pass",
            documents_present=["passport", "visa", "national_id", "driving_license", "permit"],
            passport_image=self.img,
            visa_image=self.img,
            national_id_image=self.img,
            driving_license_image=self.img,
            permit_image=self.img,
        )

        output = self.orchestrator.process_case(case_input)

        self.assertEqual(output.status, CaseStatus.COMPLETED.value)
        self.assertEqual(output.ai_generation_check, AIGateStatus.PASSED.value)
        self.assertIsNotNone(output.forensic_pass_score)
        self.assertGreaterEqual(output.forensic_pass_score, 0.0)
        self.assertLessEqual(output.forensic_pass_score, 100.0)
        self.assertEqual(
            output.documents_analyzed,
            ["passport", "visa", "national_id", "driving_license", "permit"],
        )
        self.assertEqual(len(output.document_results), 5)

    def test_risk_engine_payload_with_weights_and_points_of_failure(self):
        """Verify that score, weights_applied, and granular points_of_failure are preserved."""
        case_input = CaseInput(
            uuid="case-risk-engine-payload",
            documents_present=["passport", "visa", "national_id"],
            passport_image=self.img,
            visa_image=self.img,
            national_id_image=self.img,
        )

        output = self.orchestrator.process_case(case_input)
        out_dict = output.to_dict()

        # 1. Weights must be saved
        self.assertIn("weights_applied", out_dict)
        self.assertIn("tamper_weight", out_dict["weights_applied"])
        self.assertIn("guilloche_weight", out_dict["weights_applied"])
        self.assertAlmostEqual(
            out_dict["weights_applied"]["tamper_weight"] + out_dict["weights_applied"]["guilloche_weight"],
            1.0,
            places=2,
        )

        # 2. Score must be saved
        self.assertIn("forensic_pass_score", out_dict)
        self.assertEqual(out_dict["forensic_pass_score"], output.forensic_pass_score)

        # 3. Points of failure must be saved with specific failure checks
        self.assertIn("points_of_failure", out_dict)
        self.assertIn("failures_detected", out_dict)
        self.assertIsInstance(out_dict["points_of_failure"], list)
        for failure in out_dict["points_of_failure"]:
            self.assertIn("document", failure)
            self.assertIn("check", failure)
            self.assertIn("status", failure)
            self.assertIn("failure_reason", failure)


if __name__ == "__main__":
    unittest.main()