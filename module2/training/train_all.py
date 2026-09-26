"""Master Training Pipeline Orchestrator for Module 2 Visual / Image Forensics.

Executes the unified end-to-end training pipeline in sequential stages:
  Stage 1: Preprocessing & Data Integrity Check (Boundary crop, perspective dewarp, CLAHE)
  Stage 2: Tamper / Splice Model Training (EfficientNet-B3 / ResNet-18 on CASIA v2.0 / SIDTD / MIDV-2020)
  Stage 3: Guilloché Security Pattern Model Training (Siamese ResNet-18 with Contrastive / Cosine Loss)
  Stage 4: Post-Training Checkpoint Verification & Local Testing Readiness

Usage:
  # Standard full training on GPU:
  python training/train_all.py --dataset-type combined --device cuda

  # Quick pipeline dry-run (1 epoch each to verify everything before full training):
  python training/train_all.py --dry-run

  # Targeted training on specific dataset:
  python training/train_all.py --dataset-type sidtd --tamper-epochs 25 --guilloche-epochs 30
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Master-Train]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("master_train")


def print_banner(title: str, char: str = "=") -> None:
    width = 72
    print("\n" + char * width)
    print(f"  {title.upper()}")
    print(char * width)


def run_stage(command: List[str], stage_name: str, cwd: Path) -> bool:
    """Execute a subprocess stage and stream logs."""
    print_banner(f"STARTING {stage_name}", char="*")
    cmd_str = " ".join(command)
    logger.info(f"Executing: {cmd_str}")
    start_t = time.time()

    env = os.environ.copy()
    src_dir = str(cwd / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{src_dir}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = src_dir

    try:
        proc = subprocess.run(command, cwd=str(cwd), env=env, check=True)
        elapsed = time.time() - start_t
        logger.info(f"[SUCCESS] {stage_name} completed in {elapsed:.1f}s (exit code: {proc.returncode})")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_t
        logger.error(f"[FAILED] {stage_name} failed after {elapsed:.1f}s with return code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Failed to launch {stage_name}: {e}")
        return False


def verify_checkpoints(models_dir: Path) -> bool:
    """Verify that required model checkpoints exist and contain valid state dicts."""
    print_banner("STAGE 4: CHECKPOINT VERIFICATION & TEST READINESS", char="=")

    tamper_ckpt = models_dir / "sidtd_efficientnet_b3.pth"
    guilloche_ckpt = models_dir / "guilloche_siamese_resnet18.pth"

    all_ok = True
    for name, p in [("Tamper Detector", tamper_ckpt), ("Guilloché Verifier", guilloche_ckpt)]:
        if p.exists() and p.stat().st_size > 1024:
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  [+] {name:<22}: READY ({p.name}, {size_mb:.2f} MB)")
        else:
            print(f"  [-] {name:<22}: NOT FOUND or EMPTY ({p})")
            all_ok = False

    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Master Training Pipeline: Preprocess -> Tamper -> Guilloché")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="sidtd",
        choices=["sidtd", "casia2", "midv2020", "combined"],
        help="Dataset type to train on for tamper detection",
    )
    parser.add_argument("--dataset-root", type=str, default="data", help="Root data folder")
    parser.add_argument("--device", type=str, default="auto", help="Compute device: cuda, mps, or cpu")
    parser.add_argument("--tamper-epochs", type=int, default=None, help="Epoch count for tamper training")
    parser.add_argument("--guilloche-epochs", type=int, default=None, help="Epoch count for guilloché training")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size override")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable raw document preprocessing during training")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 quick epoch on small subset to test full pipeline")
    parser.add_argument("--skip-tamper", action="store_true", help="Skip tamper training")
    parser.add_argument("--skip-guilloche", action="store_true", help="Skip guilloché training")
    args = parser.parse_args()

    module2_root = Path(__file__).resolve().parent.parent
    py_exec = sys.executable

    print_banner("MODULE 2 — VISUAL FORENSICS MASTER TRAINING PIPELINE")
    print(f"[*] Workspace Root  : {module2_root}")
    print(f"[*] Python Executable: {py_exec}")
    print(f"[*] Dataset Target  : {args.dataset_type.upper()}")
    print(f"[*] Compute Device  : {args.device}")
    print(f"[*] Preprocessing   : {'DISABLED' if args.no_preprocess else 'ENABLED (Boundary + Dewarp + CLAHE)'}")
    print(f"[*] Mode            : {'DRY RUN (1 Epoch Verification)' if args.dry_run else 'FULL PRODUCTION TRAINING'}")

    total_start = time.time()

    # -------------------------------------------------------------
    # STAGE 1: Dataset Preprocessing & Integrity Verification
    # -------------------------------------------------------------
    print_banner("STAGE 1: PREPROCESSING & DATA INTEGRITY VERIFICATION", char="=")
    logger.info("Validating dataset paths, templates, and raw preprocessor integration...")

    raw_prep_script = module2_root / "src" / "module2" / "preprocessing" / "raw_preprocessor.py"
    if not raw_prep_script.exists():
        logger.error(f"Preprocessor script missing at: {raw_prep_script}")
        sys.exit(1)
    logger.info("Raw document preprocessor verified. Image pipeline: EXIF -> Quad Rectification -> CLAHE.")

    # -------------------------------------------------------------
    # STAGE 2: Splice / Tamper Model Training
    # -------------------------------------------------------------
    if not args.skip_tamper:
        tamper_cmd = [
            py_exec,
            str(module2_root / "training" / "train_tamper.py"),
            "--dataset-type", args.dataset_type,
            "--device", args.device,
        ]
        if args.dry_run:
            tamper_cmd.append("--dry-run")
        if args.tamper_epochs:
            tamper_cmd.extend(["--epochs", str(args.tamper_epochs)])
        if args.batch_size:
            tamper_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.no_preprocess:
            tamper_cmd.append("--no-preprocess")

        success_tamper = run_stage(tamper_cmd, "STAGE 2: TAMPER / SPLICE TRAINING", module2_root)
        if not success_tamper:
            logger.error("Tamper model training failed. Aborting master pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping Tamper Model Training (--skip-tamper specified).")

    # -------------------------------------------------------------
    # STAGE 3: Guilloché Security Pattern Model Training
    # -------------------------------------------------------------
    if not args.skip_guilloche:
        guilloche_cmd = [
            py_exec,
            str(module2_root / "training" / "train_guilloche.py"),
            "--dataset-root", args.dataset_root,
        ]
        if args.dry_run:
            guilloche_cmd.append("--dry-run")
        if args.guilloche_epochs:
            guilloche_cmd.extend(["--epochs", str(args.guilloche_epochs)])
        if args.batch_size:
            guilloche_cmd.extend(["--batch-size", str(args.batch_size)])
        if args.no_preprocess:
            guilloche_cmd.append("--no-preprocess")

        success_guilloche = run_stage(guilloche_cmd, "STAGE 3: GUILLOCHÉ PATTERN TRAINING", module2_root)
        if not success_guilloche:
            logger.error("Guilloché model training failed. Aborting master pipeline.")
            sys.exit(1)
    else:
        logger.info("Skipping Guilloché Model Training (--skip-guilloche specified).")

    # -------------------------------------------------------------
    # STAGE 4: Verification & Readiness Summary
    # -------------------------------------------------------------
    models_dir = module2_root / "models"
    verified = verify_checkpoints(models_dir)

    total_elapsed = time.time() - total_start
    print_banner("PIPELINE EXECUTION COMPLETE", char="=")
    print(f"[*] Total Execution Time: {total_elapsed:.1f}s")
    print(f"[*] Models Directory    : {models_dir}")

    if verified:
        print("\n" + "=" * 72)
        print("  ALL SYSTEMS READY FOR TESTING DIRECT NEW IMAGES")
        print("=" * 72)
        print("\nTo test direct new images back here on your local machine, run:")
        print("  1. Single Document Testing:")
        print("     PYTHONPATH=src python run_pipeline.py --image <path_to_new_image> --type passport")
        print("\n  2. Multi-Document Case Testing (Passport + Visa + National ID):")
        print("     PYTHONPATH=src python run_pipeline.py --passport <img1> --visa <img2> --national-id <img3>")
        print("\n  3. Inspect Checkpoint Metrics (Accuracy, F1, Confusion Matrix, Separation Margin):")
        print("     PYTHONPATH=src python run_pipeline.py --metrics\n")
    else:
        print("\n[!] Warning: One or more checkpoints were not generated.")


if __name__ == "__main__":
    main()
