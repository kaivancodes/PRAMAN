"""Case-level orchestrator for Module 2 — Visual / Image Forensics.

Controls case state transitions, sequential AI gate evaluation, early stop,
dual-branch forensic execution, scoring, and Risk Engine output generation.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

from module2.document.document_processor import DocumentProcessor
from module2.preprocessing.image_loader import SafeImageLoader
from module2.schemas.input_schema import CaseInput, DocumentType, ValidationError
from module2.schemas.output_schema import (
    AIGateStatus,
    CaseOutput,
    CaseStatus,
    DocumentForensicResult,
    GuillocheStatus,
    TamperStatus,
)
from module2.scoring.forensic_scorer import ForensicScorer
from module2.utils.logger import get_logger

logger = get_logger("module2.orchestrator")

DETERMINISTIC_DOCUMENT_ORDER = [
    DocumentType.PASSPORT.value,
    DocumentType.VISA.value,
    DocumentType.NATIONAL_ID.value,
    DocumentType.DRIVING_LICENSE.value,
    DocumentType.PERMIT.value,
]


class CaseState(str, Enum):
    """Case lifecycle states for Module 2."""
    RECEIVED = "RECEIVED"
    VALIDATED = "VALIDATED"
    AI_GATE_RUNNING = "AI_GATE_RUNNING"
    AI_GATE_FAILED = "AI_GATE_FAILED"
    AI_GATE_PASSED = "AI_GATE_PASSED"
    FORENSICS_RUNNING = "FORENSICS_RUNNING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


class Module2Orchestrator:
    """Controls the complete case-level lifecycle for Module 2."""

    def __init__(
        self,
        document_processor: Optional[DocumentProcessor] = None,
        scorer: Optional[ForensicScorer] = None,
        max_workers: int = 4,
    ):
        """Initialize Module 2 Orchestrator.

        Args:
            document_processor: Generic DocumentProcessor instance.
            scorer: ForensicScorer instance.
            max_workers: Concurrency worker count for forensic branches.
        """
        self.processor = document_processor or DocumentProcessor()
        self.scorer = scorer or ForensicScorer()
        self.max_workers = max_workers

    def process_case(self, case_input: CaseInput) -> CaseOutput:
        """Process an entire case through Module 2.1 and Module 2.2.

        Args:
            case_input: CaseInput object containing UUID, documents_present, and images.

        Returns:
            CaseOutput matching Risk Engine contract.
        """
        state = CaseState.RECEIVED
        logger.info(f"Case received: uuid={case_input.uuid}")

        # 1. Input Validation
        try:
            case_input.validate()
            state = CaseState.VALIDATED
            logger.info(f"Input validation passed for case {case_input.uuid}")
        except ValidationError as e:
            logger.error(f"Input validation failed for case {case_input.uuid}: {e}")
            return CaseOutput(
                uuid=case_input.uuid,
                status=CaseStatus.ERROR.value,
                ai_generation_check=AIGateStatus.FAILED.value,
                forensic_pass_score=None,
                weights_applied=self.scorer.get_weights(),
                points_of_failure=[{
                    "document": "case",
                    "check": "INPUT_VALIDATION",
                    "status": "ERROR",
                    "failure_reason": f"Validation error: {str(e)}",
                }],
                error_message=f"Validation error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Unexpected error during validation for case {case_input.uuid}: {e}")
            return CaseOutput(
                uuid=case_input.uuid,
                status=CaseStatus.ERROR.value,
                ai_generation_check=AIGateStatus.FAILED.value,
                forensic_pass_score=None,
                weights_applied=self.scorer.get_weights(),
                points_of_failure=[{
                    "document": "case",
                    "check": "INPUT_VALIDATION",
                    "status": "ERROR",
                    "failure_reason": f"Unexpected error: {str(e)}",
                }],
                error_message=f"Unexpected error: {str(e)}",
            )

        # 2. Establish deterministic document order for declared documents
        declared_set = set(d.strip().lower() for d in case_input.documents_present)
        ordered_docs = [d for d in DETERMINISTIC_DOCUMENT_ORDER if d in declared_set]

        # Decode and load images safely (preserving originals)
        loaded_images: Dict[str, Image.Image] = {}
        for doc_type in ordered_docs:
            raw_img = case_input.get_image(doc_type)
            try:
                loaded_images[doc_type] = SafeImageLoader.load_image(raw_img)
            except Exception as e:
                logger.error(f"Failed to load image for {doc_type} in case {case_input.uuid}: {e}")
                return CaseOutput(
                    uuid=case_input.uuid,
                    status=CaseStatus.ERROR.value,
                    ai_generation_check=AIGateStatus.FAILED.value,
                    forensic_pass_score=None,
                    weights_applied=self.scorer.get_weights(),
                    points_of_failure=[{
                        "document": doc_type,
                        "check": "IMAGE_LOADING",
                        "status": "ERROR",
                        "failure_reason": f"Image loading error for {doc_type}: {str(e)}",
                    }],
                    error_message=f"Image loading error for {doc_type}: {str(e)}",
                )

        # 3. Module 2.1: Sequential AI Gate
        state = CaseState.AI_GATE_RUNNING
        document_results: Dict[str, Dict[str, Any]] = {}
        flagged_document: Optional[str] = None

        for doc_type in ordered_docs:
            img = loaded_images[doc_type]
            is_ai_generated = self.processor.run_ai_gate(img, doc_type=doc_type)

            if is_ai_generated:
                flagged_document = doc_type
                state = CaseState.AI_GATE_FAILED
                logger.warning(
                    f"AI-generated document detected: {doc_type} in case {case_input.uuid}. "
                    "Raising RED FLAG and stopping Module 2 workflow immediately."
                )

                # Record result for the flagged document
                document_results[doc_type] = DocumentForensicResult(
                    document_type=doc_type,
                    ai_test=AIGateStatus.FAIL.value,
                    ai_gate_status=AIGateStatus.FAIL.value,
                    ai_generated=True,
                ).to_dict()

                # STOP IMMEDIATELY: Do NOT process subsequent documents or run Module 2.2
                return CaseOutput(
                    uuid=case_input.uuid,
                    status=CaseStatus.RED_FLAG.value,
                    ai_test=AIGateStatus.FAIL.value,
                    ai_generation_check=AIGateStatus.FAIL.value,
                    flag_type="AI_GENERATED_DOCUMENT",
                    flagged_document=flagged_document,
                    forensic_pass_score=None,
                    weights_applied=self.scorer.get_weights(),
                    points_of_failure=[{
                        "document": flagged_document,
                        "check": "AI_TEST",
                        "status": AIGateStatus.FAIL.value,
                        "failure_reason": (
                            f"AI Test failed on '{flagged_document}'. Document flagged as AI-generated / synthetic. "
                            "Module 2 workflow halted immediately."
                        ),
                    }],
                    documents_analyzed=[doc_type],
                    document_results=document_results,
                )

            # Passed AI gate for this document
            document_results[doc_type] = DocumentForensicResult(
                document_type=doc_type,
                ai_test=AIGateStatus.PASS.value,
                ai_gate_status=AIGateStatus.PASS.value,
                ai_generated=False,
            ).to_dict()



        # All documents in documents_present have passed the AI gate
        state = CaseState.AI_GATE_PASSED
        logger.info(f"All {len(ordered_docs)} documents passed AI gate for case {case_input.uuid}")

        # 4. Module 2.2: Visual / Image Forensics
        state = CaseState.FORENSICS_RUNNING
        logger.info(f"Module 2.2 started for case {case_input.uuid}")

        doc_scoring_summaries: List[Dict[str, Any]] = []

        # Execute Module 2.2 for each document (using thread pool where safe)
        def _process_doc_forensics(dtype: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
            doc_img = loaded_images[dtype]
            # Metadata lookup for country/version if provided in case_input.metadata
            doc_meta = case_input.metadata.get(dtype, {})
            country = doc_meta.get("country", case_input.metadata.get("country"))
            version = doc_meta.get("version", "default")
            region = doc_meta.get("region", "background")
            is_applicable = doc_meta.get("guilloche_applicable", True)

            tamper_res, guilloche_res, _ = self.processor.run_forensics(
                original_image=doc_img,
                doc_type=dtype,
                country=country,
                version=version,
                region=region,
                is_guilloche_applicable=is_applicable,
            )
            return dtype, tamper_res, guilloche_res

        # Concurrency safety: Apple MPS does not support concurrent forward passes across threads
        is_mps = getattr(self.processor.device, "type", "") == "mps"
        effective_workers = 1 if is_mps else self.max_workers

        if len(ordered_docs) > 1 and effective_workers > 1:
            logger.debug(f"Running forensic branches concurrently with {effective_workers} workers")
            with ThreadPoolExecutor(max_workers=min(effective_workers, len(ordered_docs))) as executor:
                futures = {
                    executor.submit(_process_doc_forensics, dtype): dtype
                    for dtype in ordered_docs
                }
                for future in as_completed(futures):
                    dtype = futures[future]
                    try:
                        _, tamper_res, guilloche_res = future.result()
                        doc_score_dict = self.scorer.score_document(
                            tamper_res, guilloche_res, doc_type=dtype
                        )
                        doc_scoring_summaries.append(doc_score_dict)

                        # Update internal document results
                        document_results[dtype]["tamper_result"] = tamper_res
                        document_results[dtype]["guilloche_result"] = guilloche_res
                        document_results[dtype]["forensic_score"] = doc_score_dict["forensic_pass_score"]
                    except Exception as e:
                        logger.error(f"Error executing forensics for {dtype}: {e}")
                        tamper_error = {"status": TamperStatus.ERROR.value, "passed": False, "error": str(e)}
                        guilloche_error = {"status": GuillocheStatus.ERROR.value, "passed": False, "error": str(e)}
                        doc_score_dict = self.scorer.score_document(
                            tamper_error, guilloche_error, doc_type=dtype
                        )
                        doc_scoring_summaries.append(doc_score_dict)
                        document_results[dtype]["tamper_result"] = tamper_error
                        document_results[dtype]["guilloche_result"] = guilloche_error
                        document_results[dtype]["forensic_score"] = 0.0
        else:
            logger.debug("Executing document forensic branches sequentially")
            for dtype in ordered_docs:
                try:
                    _, tamper_res, guilloche_res = _process_doc_forensics(dtype)
                    doc_score_dict = self.scorer.score_document(
                        tamper_res, guilloche_res, doc_type=dtype
                    )
                    doc_scoring_summaries.append(doc_score_dict)

                    document_results[dtype]["tamper_result"] = tamper_res
                    document_results[dtype]["guilloche_result"] = guilloche_res
                    document_results[dtype]["forensic_score"] = doc_score_dict["forensic_pass_score"]
                except Exception as e:
                    logger.error(f"Error executing forensics for {dtype}: {e}")
                    tamper_error = {"status": TamperStatus.ERROR.value, "passed": False, "error": str(e)}
                    guilloche_error = {"status": GuillocheStatus.ERROR.value, "passed": False, "error": str(e)}
                    doc_score_dict = self.scorer.score_document(
                        tamper_error, guilloche_error, doc_type=dtype
                    )
                    doc_scoring_summaries.append(doc_score_dict)
                    document_results[dtype]["tamper_result"] = tamper_error
                    document_results[dtype]["guilloche_result"] = guilloche_error
                    document_results[dtype]["forensic_score"] = 0.0

        # 5. Final Forensic Pass Score Calculation
        final_score = self.scorer.score_case(doc_scoring_summaries)
        state = CaseState.COMPLETED
        logger.info(
            f"Final score calculated for case {case_input.uuid}: "
            f"forensic_pass_score = {final_score}/100"
        )
        logger.info(f"Case completed: uuid={case_input.uuid}")

        # 6. Detailed Point-of-Failure Aggregation
        points_of_failure: List[Dict[str, Any]] = []
        for dtype in ordered_docs:
            doc_res = document_results.get(dtype, {})
            tamper_res = doc_res.get("tamper_result") or {}
            guilloche_res = doc_res.get("guilloche_result") or {}

            # Tamper / Forgery check evaluation
            if not tamper_res.get("passed", False):
                verdict = tamper_res.get("label", TamperStatus.FORGED.value)
                forgery_prob = tamper_res.get("forgery_probability")
                points_of_failure.append({
                    "document": dtype,
                    "check": "TAMPER_SPLICE",
                    "status": tamper_res.get("status", TamperStatus.FORGED.value),
                    "failure_reason": (
                        f"Tamper/forgery check failed on '{dtype}'. "
                        f"Verdict: {verdict}, Forgery Probability: {forgery_prob if forgery_prob is not None else 'N/A'}"
                    ),
                    "details": {
                        "forgery_probability": forgery_prob,
                        "confidence": tamper_res.get("confidence"),
                        "ela_mean": tamper_res.get("ela_mean"),
                        "dct_high_freq_ratio": tamper_res.get("dct_high_freq_ratio"),
                    },
                })

            # Guilloché / Background pattern check evaluation
            g_status = guilloche_res.get("status")
            if g_status == GuillocheStatus.INCONSISTENT.value:
                sim = guilloche_res.get("similarity_score")
                thresh = guilloche_res.get("threshold", 0.75)
                points_of_failure.append({
                    "document": dtype,
                    "check": "GUILLOCHE_PATTERN",
                    "status": GuillocheStatus.INCONSISTENT.value,
                    "failure_reason": (
                        f"Guilloché background pattern inconsistent with authentic reference on '{dtype}'. "
                        f"Similarity: {sim if sim is not None else 'N/A'}, Required Threshold: {thresh}"
                    ),
                    "details": {
                        "similarity_score": sim,
                        "threshold": thresh,
                        "reference_available": True,
                    },
                })
            elif (
                g_status == GuillocheStatus.REFERENCE_REQUIRED.value
                and self.scorer.missing_ref_policy == "STRICT_FAIL"
            ):
                points_of_failure.append({
                    "document": dtype,
                    "check": "GUILLOCHE_PATTERN",
                    "status": GuillocheStatus.REFERENCE_REQUIRED.value,
                    "failure_reason": (
                        f"Authentic reference template missing for '{dtype}' under STRICT_FAIL policy."
                    ),
                    "details": {
                        "reference_available": False,
                        "policy": "STRICT_FAIL",
                    },
                })
            elif g_status == GuillocheStatus.ERROR.value:
                points_of_failure.append({
                    "document": dtype,
                    "check": "GUILLOCHE_PATTERN",
                    "status": GuillocheStatus.ERROR.value,
                    "failure_reason": f"Guilloché analysis error on '{dtype}': {guilloche_res.get('error', 'unknown')}",
                })

        return CaseOutput(
            uuid=case_input.uuid,
            status=CaseStatus.COMPLETED.value,
            ai_test=AIGateStatus.PASS.value,
            ai_generation_check=AIGateStatus.PASS.value,
            forensic_pass_score=final_score,
            weights_applied=self.scorer.get_weights(),
            points_of_failure=points_of_failure,
            documents_analyzed=ordered_docs,
            document_results=document_results,
        )


