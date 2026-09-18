"""Unit tests for Module 2 input contract and validation."""

import unittest
from PIL import Image

from module2.schemas.input_schema import CaseInput, ValidationError


class TestInputContract(unittest.TestCase):
    """Test suite verifying Module 2 input specification and validation rules."""

    def setUp(self):
        self.dummy_img = Image.new("RGB", (100, 100), color=(200, 200, 200))

    def test_valid_3_document_case(self):
        """Case with required 3 documents: passport, visa, national_id."""
        case = CaseInput(
            uuid="case-3-docs-001",
            documents_present=["passport", "visa", "national_id"],
            passport_image=self.dummy_img,
            visa_image=self.dummy_img,
            national_id_image=self.dummy_img,
        )
        # Should validate without error
        case.validate()
        self.assertEqual(len(case.documents_present), 3)

    def test_valid_4_document_case_with_driving_license(self):
        """Case with 4 documents including optional driving_license."""
        case = CaseInput(
            uuid="case-4-docs-dl-002",
            documents_present=["passport", "visa", "national_id", "driving_license"],
            passport_image=self.dummy_img,
            visa_image=self.dummy_img,
            national_id_image=self.dummy_img,
            driving_license_image=self.dummy_img,
        )
        case.validate()
        self.assertEqual(len(case.documents_present), 4)

    def test_valid_4_document_case_with_permit(self):
        """Case with 4 documents including optional permit."""
        case = CaseInput(
            uuid="case-4-docs-permit-003",
            documents_present=["passport", "visa", "national_id", "permit"],
            passport_image=self.dummy_img,
            visa_image=self.dummy_img,
            national_id_image=self.dummy_img,
            permit_image=self.dummy_img,
        )
        case.validate()
        self.assertEqual(len(case.documents_present), 4)

    def test_valid_5_document_case(self):
        """Case with all 5 documents."""
        case = CaseInput(
            uuid="case-5-docs-004",
            documents_present=["passport", "visa", "national_id", "driving_license", "permit"],
            passport_image=self.dummy_img,
            visa_image=self.dummy_img,
            national_id_image=self.dummy_img,
            driving_license_image=self.dummy_img,
            permit_image=self.dummy_img,
        )
        case.validate()
        self.assertEqual(len(case.documents_present), 5)

    def test_missing_required_document(self):
        """Missing national_id must fail validation."""
        case = CaseInput(
            uuid="case-missing-req-005",
            documents_present=["passport", "visa"],
            passport_image=self.dummy_img,
            visa_image=self.dummy_img,
        )
        with self.assertRaises(ValidationError) as ctx:
            case.validate()
        self.assertIn("national_id", str(ctx.exception).lower())

    def test_invalid_document_type(self):
        """Invalid document type name must be rejected."""
        case = CaseInput(
            uuid="case-invalid-name-006",
            documents_present=["passport", "visa", "national_id", "utility_bill"],
            passport_image=self.dummy_img,
            visa_image=self.dummy_img,
            national_id_image=self.dummy_img,
        )
        with self.assertRaises(ValidationError) as ctx:
            case.validate()
        self.assertIn("utility_bill", str(ctx.exception))

    def test_optional_document_absent_is_allowed(self):
        """Permit absent from documents_present should not cause issues or wait."""
        case = CaseInput(
            uuid="case-optional-absent-007",
            documents_present=["passport", "visa", "national_id", "driving_license"],
            passport_image=self.dummy_img,
            visa_image=self.dummy_img,
            national_id_image=self.dummy_img,
            driving_license_image=self.dummy_img,
            permit_image=None,  # explicitly None
        )
        case.validate()
        self.assertIsNone(case.get_image("permit"))

    def test_declared_document_missing_image(self):
        """If document is listed in documents_present but image is None, fail validation."""
        case = CaseInput(
            uuid="case-missing-img-008",
            documents_present=["passport", "visa", "national_id"],
            passport_image=self.dummy_img,
            visa_image=self.dummy_img,
            national_id_image=None,  # missing image!
        )
        with self.assertRaises(ValidationError) as ctx:
            case.validate()
        self.assertIn("national_id_image is None", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
