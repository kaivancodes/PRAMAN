"""Unit tests for Forensic Scorer.

Verifies weighted pass scoring combinations and edge cases:
- PASS + PASS = 100
- PASS + FAIL = 60
- FAIL + PASS = 40
- FAIL + FAIL = 0
- Handling of REFERENCE_REQUIRED and NOT_APPLICABLE.
"""

import unittest
from module2.schemas.output_schema import GuillocheStatus, TamperStatus
from module2.scoring.forensic_scorer import ForensicScorer


class TestForensicScorer(unittest.TestCase):
    """Test suite for forensic pass score calculation."""

    def setUp(self):
        # Default 60/40 weights
        self.scorer = ForensicScorer(
            tamper_weight=0.60,
            guilloche_weight=0.40,
            missing_ref_policy="REWEIGHT",
        )

    def test_pass_and_pass_yields_100(self):
        """Tamper = PASS (60%) + Guilloche = PASS (40%) -> 100."""
        tamper = {"status": TamperStatus.BONA_FIDE.value, "passed": True}
        guilloche = {"status": GuillocheStatus.CONSISTENT.value, "passed": True}

        res = self.scorer.score_document(tamper, guilloche, doc_type="passport")
        self.assertEqual(res["forensic_pass_score"], 100.0)

    def test_pass_and_fail_yields_60(self):
        """Tamper = PASS (60%) + Guilloche = FAIL (0%) -> 60."""
        tamper = {"status": TamperStatus.BONA_FIDE.value, "passed": True}
        guilloche = {"status": GuillocheStatus.INCONSISTENT.value, "passed": False}

        res = self.scorer.score_document(tamper, guilloche, doc_type="passport")
        self.assertEqual(res["forensic_pass_score"], 60.0)

    def test_fail_and_pass_yields_40(self):
        """Tamper = FAIL (0%) + Guilloche = PASS (40%) -> 40."""
        tamper = {"status": TamperStatus.FORGED.value, "passed": False}
        guilloche = {"status": GuillocheStatus.CONSISTENT.value, "passed": True}

        res = self.scorer.score_document(tamper, guilloche, doc_type="passport")
        self.assertEqual(res["forensic_pass_score"], 40.0)

    def test_fail_and_fail_yields_0(self):
        """Tamper = FAIL (0%) + Guilloche = FAIL (0%) -> 0."""
        tamper = {"status": TamperStatus.FORGED.value, "passed": False}
        guilloche = {"status": GuillocheStatus.INCONSISTENT.value, "passed": False}

        res = self.scorer.score_document(tamper, guilloche, doc_type="passport")
        self.assertEqual(res["forensic_pass_score"], 0.0)

    def test_reference_required_with_reweight_policy(self):
        """Under REWEIGHT, missing Guilloché reference normalizes to Tamper check."""
        tamper_pass = {"status": TamperStatus.BONA_FIDE.value, "passed": True}
        tamper_fail = {"status": TamperStatus.FORGED.value, "passed": False}
        guilloche_ref_req = {"status": GuillocheStatus.REFERENCE_REQUIRED.value, "passed": None}

        # If tamper passed and guilloche is missing reference -> 100% of applicable checks passed
        res_pass = self.scorer.score_document(tamper_pass, guilloche_ref_req, doc_type="visa")
        self.assertEqual(res_pass["forensic_pass_score"], 100.0)

        # If tamper failed and guilloche is missing reference -> 0% of applicable checks passed
        res_fail = self.scorer.score_document(tamper_fail, guilloche_ref_req, doc_type="visa")
        self.assertEqual(res_fail["forensic_pass_score"], 0.0)

    def test_reference_required_with_strict_fail_policy(self):
        """Under STRICT_FAIL, missing Guilloché reference counts as unpassed (0/40)."""
        scorer_strict = ForensicScorer(
            tamper_weight=0.60,
            guilloche_weight=0.40,
            missing_ref_policy="STRICT_FAIL",
        )
        tamper_pass = {"status": TamperStatus.BONA_FIDE.value, "passed": True}
        guilloche_ref_req = {"status": GuillocheStatus.REFERENCE_REQUIRED.value, "passed": None}

        res = scorer_strict.score_document(tamper_pass, guilloche_ref_req, doc_type="visa")
        self.assertEqual(res["forensic_pass_score"], 60.0)

    def test_case_aggregation(self):
        """Case pass score is the mean across evaluated documents."""
        doc1 = {"forensic_pass_score": 100.0}
        doc2 = {"forensic_pass_score": 60.0}
        doc3 = {"forensic_pass_score": 80.0}

        case_score = self.scorer.score_case([doc1, doc2, doc3])
        self.assertEqual(case_score, 80.0)

    def test_custom_weights(self):
        """Custom configurable weights (e.g. 70/30)."""
        scorer_custom = ForensicScorer(tamper_weight=0.70, guilloche_weight=0.30)
        tamper = {"status": TamperStatus.BONA_FIDE.value, "passed": True}
        guilloche = {"status": GuillocheStatus.INCONSISTENT.value, "passed": False}

        res = scorer_custom.score_document(tamper, guilloche)
        self.assertEqual(res["forensic_pass_score"], 70.0)


if __name__ == "__main__":
    unittest.main()
