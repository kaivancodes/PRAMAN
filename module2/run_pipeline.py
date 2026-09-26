"""CLI for running the COMPLETE Module 2 Visual / Image Forensics Pipeline.

Runs the full end-to-end architecture:
1. Input Validation & Document Registry.
2. Raw Document Preprocessing:
   - EXIF auto-orientation (corrects camera tilt)
   - 4-point quadrilateral detection & perspective homography warp
   - Saliency / Sobel energy bounding box crop (auto-removes desk/background)
   - CLAHE illumination normalization (LAB color space shadow equalization)
3. Module 2.1: Sequential AI-Generated Image Detection Gate (Deterministic order).
4. Module 2.2: Visual Forensics (Dual Branch):
   - Multi-modal Forensic Preprocessor (ELA + DCT feature extraction)
   - Splice / Tamper Detection (EfficientNet-B3 with forensic context)
   - Guilloché Security Pattern Verification (Reference template similarity)
5. Forensic Pass Scorer (Default 60/40 weighted fusion).
6. Generates SIH-compliant structured JSON output for Risk Engine.

Usage Examples:
  # 1. Run full pipeline on a single raw image (passport/ID):
  python run_pipeline.py --image my_id_4.jpeg --type passport

  # 2. Run full pipeline on a complete multi-document case:
  python run_pipeline.py --passport my_id_4.jpeg --visa data/templates/Images/reals/aze_passport_00.jpg --national-id data/templates/Images/reals/esp_id_00.jpg

  # 3. Inspect trained checkpoint metrics:
  python run_pipeline.py --metrics

  # 4. Verify on benchmark sample:
  python run_pipeline.py --verify-sample

  # 5. Simulate an AI gate RED FLAG (e.g. AI-generated passport stops the entire case):
  python run_pipeline.py --image my_id_3.png --type passport --simulate-ai
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from PIL import Image

# Ensure src/ is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from module2.document.document_processor import DocumentProcessor, ModularAIDetector
from module2.guilloche.guilloche_detector import GuillocheDetector
from module2.guilloche.similarity import GuillocheSimilarityEvaluator
from module2.orchestrator.module2_orchestrator import Module2Orchestrator
from module2.preprocessing.forensic_preprocessor import ForensicPreprocessor
from module2.schemas.input_schema import CaseInput
from module2.scoring.forensic_scorer import ForensicScorer
from module2.tamper.tamper_detector import TamperDetector


def print_banner(title: str):
    print("\n" + "=" * 68)
    print(f"  {title.upper()}")
    print("=" * 68)


def print_section(title: str):
    print("\n" + "-" * 68)
    print(f"  {title}")
    print("-" * 68)


def show_checkpoint_metrics(checkpoint_path: str = "models/sidtd_efficientnet_b3.pth"):
    """Inspect and display metadata, loss function, accuracy, and confusion matrix from checkpoint."""
    import torch
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        print(f"[!] Checkpoint not found at: {ckpt_p}")
        return

    print_banner(f"Model Checkpoint Inspection: {ckpt_p.name}")
    try:
        checkpoint = torch.load(ckpt_p, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_p, map_location="cpu")

    epoch = checkpoint.get("epoch", "N/A")
    model_name = checkpoint.get("model_name", "efficientnet_b3")
    classes = checkpoint.get("class_names", ["BONA_FIDE", "FORGED"])
    val_metrics = checkpoint.get("val_metrics", {})
    training_config = checkpoint.get("training_config", {})

    print(f"[*] Architecture       : {model_name.upper()}")
    print(f"[*] Checkpoint Epoch   : {epoch}")
    print(f"[*] Classes            : {classes}")

    if training_config:
        print_section("TRAINING CONFIGURATION")
        for k, v in training_config.items():
            print(f"  * {k:<22}: {v}")

    if val_metrics:
        print_section("VALIDATION METRICS")
        for k, v in val_metrics.items():
            if k == "confusion_matrix":
                cm = v
                print(f"  * Confusion Matrix     : {cm}")
                if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
                    print(f"      - True Real (TN)   : {cm[0][0]}")
                    print(f"      - False Forged (FP): {cm[0][1]}")
                    print(f"      - False Real (FN)  : {cm[1][0]}")
                    print(f"      - True Forged (TP) : {cm[1][1]}")
            elif isinstance(v, (int, float)):
                print(f"  * {k.replace('_', ' ').title():<22}: {v:.4f}")
            else:
                print(f"  * {k.replace('_', ' ').title():<22}: {v}")
    elif "loss_function" in checkpoint:
        print_section("CHECKPOINT METADATA")
        print(f"  * Loss Function        : {checkpoint.get('loss_function', {}).get('name')}")
        if "confusion_matrix" in checkpoint:
            print(f"  * Confusion Matrix     : {checkpoint.get('confusion_matrix')}")

    print_banner("Inspection Complete")


def run_full_pipeline(
    passport_path: Path = None,
    visa_path: Path = None,
    national_id_path: Path = None,
    driving_license_path: Path = None,
    permit_path: Path = None,
    checkpoint_path: str = "models/sidtd_efficientnet_b3.pth",
    simulate_ai_flag: str = None,
    device: str = "auto",
    output_json_path: str = None,
    reference_path: Path = None,
    country: str = "IND",
):
    print_banner("Module 2 — Visual / Image Forensics Pipeline")

    # 1. Initialize detector components
    print(f"[*] Initializing EfficientNet-B3 Tamper Detector ({checkpoint_path})...")
    tamper_det = TamperDetector(checkpoint_path=checkpoint_path, device=device)
    
    ai_det = ModularAIDetector()
    ai_gate_enabled = bool(simulate_ai_flag)  # Only enable AI gate check if actively testing simulated AI flag
    if simulate_ai_flag:
        ai_det.set_flagged_for_testing(simulate_ai_flag.lower())
        print(f"[!] Simulation mode: {simulate_ai_flag.upper()} is flagged as AI-GENERATED (Module 2.1 gate test)")
    else:
        print("[*] Module 2.1 AI Gate: BYPASSED (AI generation check is completed/done). Proceeding to Module 2.2.")

    guilloche_ckpt = Path("models/guilloche_siamese_resnet18.pth")
    if guilloche_ckpt.exists():
        print(f"[*] Initializing Siamese ResNet-18 Guilloché Pattern Verifier ({guilloche_ckpt})...")
        evaluator = GuillocheSimilarityEvaluator(
            checkpoint_path=str(guilloche_ckpt),
            model_name="resnet18",
            device=device,
        )
        guilloche_det = GuillocheDetector(similarity_evaluator=evaluator)
    else:
        evaluator = GuillocheSimilarityEvaluator(
            model_name="resnet18",
            device=device,
        )
        guilloche_det = GuillocheDetector(similarity_evaluator=evaluator)

    if reference_path and reference_path.exists():
        ref_img = Image.open(reference_path).convert("RGB")
        print(f"[*] Registered authentic Guilloché reference template: {reference_path.name} (country: {country})")
        for d in ("passport", "visa", "national_id", "driving_license", "permit"):
            guilloche_det.reference_manager.register_reference(
                image=ref_img,
                country=country,
                document_type=d,
                version="default",
                region="background",
            )

    forensic_prep = ForensicPreprocessor()

    doc_processor = DocumentProcessor(
        ai_detector=ai_det,
        tamper_detector=tamper_det,
        guilloche_detector=guilloche_det,
        forensic_preprocessor=forensic_prep,
        device=device,
    )

    orchestrator = Module2Orchestrator(
        document_processor=doc_processor,
        scorer=ForensicScorer(),
        max_workers=1,
        ai_gate_enabled=ai_gate_enabled,
    )

    # 2. Build CaseInput
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    docs_present = []
    images_dict = {}

    if passport_path and passport_path.exists():
        docs_present.append("passport")
        images_dict["passport_image"] = Image.open(passport_path).convert("RGB")
    if visa_path and visa_path.exists():
        docs_present.append("visa")
        images_dict["visa_image"] = Image.open(visa_path).convert("RGB")
    if national_id_path and national_id_path.exists():
        docs_present.append("national_id")
        images_dict["national_id_image"] = Image.open(national_id_path).convert("RGB")
    if driving_license_path and driving_license_path.exists():
        docs_present.append("driving_license")
        images_dict["driving_license_image"] = Image.open(driving_license_path).convert("RGB")
    if permit_path and permit_path.exists():
        docs_present.append("permit")
        images_dict["permit_image"] = Image.open(permit_path).convert("RGB")

    is_single_doc = len(docs_present) == 1
    if is_single_doc:
        print(f"[*] Mode              : Individual Document Testing ('{docs_present[0]}')")
    else:
        print(f"[*] Mode              : Multi-Document Case Evaluation ({len(docs_present)} documents)")

    case_meta = {"country": country} if reference_path else {}
    case_input = CaseInput(
        uuid=case_id,
        documents_present=docs_present,
        allow_single_document=is_single_doc,
        metadata=case_meta,
        **images_dict,
    )

    print(f"[*] Case ID           : {case_id}")
    print(f"[*] Documents Present : {docs_present}")

    # 3. Process Case through the whole pipeline
    output = orchestrator.process_case(case_input)

    # 4. Display human-readable results
    print_section("STEP 1: MODULE 2.1 — AI GENERATION GATE")
    ai_status = output.ai_generation_check
    if ai_status in ("PASS", "PASSED"):
        print(f"  Result           : [PASSED] (All documents verified as authentic / non-AI)")
    else:
        print(f"  Result           : [RED FLAG] (Synthetic / AI-Generated document detected!)")
        print(f"  Flagged Document : {output.flagged_document}")
        print(f"  Action           : Pipeline terminated immediately. Forensics aborted.")

    print_section("STEP 2: MODULE 2.2 — VISUAL FORENSICS BREAKDOWN")
    if output.status == "RED_FLAG":
        print("  Forensics Aborted: Module 2.2 was skipped due to RED FLAG at Module 2.1 AI Gate.")
    else:
        for doc_name, res in output.document_results.items():
            print(f"\n  Document: [{doc_name.upper()}]")
            
            # Raw Preprocessing Metadata
            prep = res.get("preprocessing_metadata") or {}
            if prep:
                orig_sz = prep.get("original_size", ("?", "?"))
                fin_sz = prep.get("final_size", ("?", "?"))
                cropped = prep.get("boundary_cropped", False)
                rectified = prep.get("perspective_rectified", False)
                clahe = prep.get("clahe_applied", False)
                print(f"    - Raw Preprocessing           : Input {orig_sz[0]}x{orig_sz[1]} -> Output {fin_sz[0]}x{fin_sz[1]} "
                      f"[Desk Cropped: {cropped}, Dewarped: {rectified}, CLAHE: {clahe}]")

            # Tamper results
            t_res = res.get("tamper_result") or {}
            t_status = t_res.get("label", "N/A")
            t_conf = t_res.get("confidence", 0.0) * 100
            t_fake_p = t_res.get("forgery_probability", 0.0) * 100
            print(f"    - Tamper / Splice Forensics   : [{t_status}] (Confidence: {t_conf:.2f}%, Forgery Prob: {t_fake_p:.2f}%)")
            
            # Forensic Context
            ctx = t_res.get("forensic_context") or {}
            if ctx:
                ela_m = ctx.get("ela_mean", 0.0)
                dct_e = ctx.get("dct_features", {}).get("high_freq_ratio", 0.0)
                print(f"      * ELA Compression Delta     : {ela_m:.4f}")
                print(f"      * DCT High-Freq Ratio       : {dct_e:.4f}")

            # Guilloche results
            g_res = res.get("guilloche_result") or {}
            g_status = g_res.get("status", "N/A")
            g_sim = g_res.get("similarity_score")
            sim_str = f"{g_sim * 100:.2f}%" if g_sim is not None else "N/A"
            print(f"    - Guilloché Pattern Forensics : [{g_status}] (Reference Similarity: {sim_str})")

            # Document combined score
            doc_score = res.get("forensic_score")
            if doc_score is None:
                doc_score = res.get("document_forensic_score")
            score_str = f"{doc_score:.2f} / 100.0" if doc_score is not None else "N/A"
            print(f"    => Combined Document Score    : {score_str}")

    print_section("FINAL CASE DECISION (RISK ENGINE CONTRACT)")
    print(f"  Case Status          : {output.status}")
    if output.forensic_pass_score is not None:
        print(f"  Forensic Pass Score  : {output.forensic_pass_score:.2f} / 100.0")
    else:
        print(f"  Forensic Pass Score  : null (due to RED FLAG)")
    print(f"  Weights Applied      : Tamper={output.weights_applied.get('tamper_weight')}, Guilloché={output.weights_applied.get('guilloche_weight')}")

    if output.points_of_failure:
        print(f"  Points of Failure    : {len(output.points_of_failure)} issue(s) detected")
        for pof in output.points_of_failure:
            print(f"    * [{pof.get('document')}] {pof.get('check')}: {pof.get('failure_reason')}")

    print_banner("Case Execution Complete")

    # Export to JSON
    if output_json_path:
        out_file = Path(output_json_path)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(output.to_dict(), f, indent=2, default=str)
        print(f"\n[+] Full Risk Engine JSON saved to: {out_file.resolve()}\n")

    return output


def main():
    parser = argparse.ArgumentParser(description="Run Full Module 2 Visual Forensics Pipeline")
    parser.add_argument("--image", type=str, default=None, help="Path to single document to evaluate")
    parser.add_argument("--type", type=str, default="passport", choices=["passport", "visa", "national_id", "driving_license", "permit"])
    parser.add_argument("--passport", type=str, default=None, help="Path to passport image")
    parser.add_argument("--visa", type=str, default=None, help="Path to visa image")
    parser.add_argument("--national-id", type=str, default=None, help="Path to national ID image")
    parser.add_argument("--driving-license", type=str, default=None, help="Path to driving license image")
    parser.add_argument("--permit", type=str, default=None, help="Path to permit image")
    parser.add_argument("--checkpoint", type=str, default="models/sidtd_efficientnet_b3.pth", help="Path to .pth checkpoint")
    parser.add_argument("--simulate-ai", type=str, default=None, help="Document name to trigger AI gate failure (e.g. passport)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cpu, mps, cuda)")
    parser.add_argument("--reference", type=str, default=None, help="Path to authentic reference template for Guilloché pattern comparison")
    parser.add_argument("--country", type=str, default="IND", help="Issuing country code for Guilloché reference (default: IND)")
    parser.add_argument("--output", type=str, default="case_output.json", help="Path to save output JSON")
    parser.add_argument("--metrics", action="store_true", help="Display trained model checkpoint metadata and metrics")
    parser.add_argument("--verify-sample", action="store_true", help="Run quick verification against real benchmark sample")
    args = parser.parse_args()

    ref_p = Path(args.reference) if args.reference else None

    if args.metrics:
        show_checkpoint_metrics(args.checkpoint)
        guilloche_ckpt = Path("models/guilloche_siamese_resnet18.pth")
        if guilloche_ckpt.exists():
            show_checkpoint_metrics(str(guilloche_ckpt))
        return

    if args.verify_sample:
        sample_path = Path("data/templates/Images/reals/aze_passport_00.jpg")
        if not sample_path.exists():
            print(f"[!] Benchmark sample not found at: {sample_path}")
            return
        print(f"[*] Running verification pipeline on benchmark sample: {sample_path}...")
        run_full_pipeline(
            passport_path=sample_path,
            checkpoint_path=args.checkpoint,
            device=args.device,
            output_json_path=args.output,
            reference_path=ref_p,
            country=args.country,
        )
        return

    # If single image provided
    if args.image:
        img_p = Path(args.image)
        if not img_p.exists():
            print(f"[!] Image not found: {img_p}")
            return
        kwargs = {f"{args.type}_path": img_p}
        run_full_pipeline(
            checkpoint_path=args.checkpoint,
            simulate_ai_flag=args.simulate_ai,
            device=args.device,
            output_json_path=args.output,
            reference_path=ref_p,
            country=args.country,
            **kwargs,
        )
    elif args.passport or args.visa or args.national_id:
        p_p = Path(args.passport) if args.passport else None
        v_p = Path(args.visa) if args.visa else None
        n_p = Path(args.national_id) if args.national_id else None
        d_p = Path(args.driving_license) if args.driving_license else None
        m_p = Path(args.permit) if args.permit else None
        run_full_pipeline(
            passport_path=p_p,
            visa_path=v_p,
            national_id_path=n_p,
            driving_license_path=d_p,
            permit_path=m_p,
            checkpoint_path=args.checkpoint,
            simulate_ai_flag=args.simulate_ai,
            device=args.device,
            output_json_path=args.output,
            reference_path=ref_p,
            country=args.country,
        )
    else:
        print("[!] Please provide either --image <path> or --passport <path> --visa <path> --national-id <path>")
        print("\nExamples:")
        print("  python run_pipeline.py --image my_id_4.jpeg --type passport")
        print("  python run_pipeline.py --metrics")
        print("  python run_pipeline.py --verify-sample")


if __name__ == "__main__":
    main()
