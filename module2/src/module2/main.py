"""Production entry point for Module 2 — Visual / Image Forensics.

Accepts Module 2 case input, validates, invokes orchestrator, and returns
structured Risk Engine output.

Contains no CNN, ELA, DCT, scoring, or similarity algorithms directly.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from module2.document.document_processor import DocumentProcessor, ModularAIDetector
from module2.guilloche.guilloche_detector import GuillocheDetector
from module2.guilloche.reference_manager import GuillocheReferenceManager
from module2.models.efficientnet_detector import SIDTDEfficientNetDetector
from module2.orchestrator.module2_orchestrator import Module2Orchestrator
from module2.preprocessing.forensic_preprocessor import ForensicPreprocessor
from module2.schemas.input_schema import CaseInput
from module2.schemas.output_schema import CaseOutput
from module2.scoring.forensic_scorer import ForensicScorer
from module2.tamper.tamper_detector import TamperDetector
from module2.utils.device import get_device
from module2.utils.logger import get_logger

logger = get_logger("module2.main")


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load YAML configuration or return default settings."""
    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

    # Default fallback path relative to package
    default_path = Path(__file__).resolve().parent.parent.parent / "config" / "module2_config.yaml"
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    return {}


def build_orchestrator(config: Optional[Dict[str, Any]] = None) -> Module2Orchestrator:
    """Factory function to build a production Module2Orchestrator from config.

    Ensures models are loaded once.
    """
    cfg = config or load_config()

    device_pref = cfg.get("runtime", {}).get("device", "auto")
    device = get_device(device_pref)

    # 1. Models & Detectors
    models_cfg = cfg.get("models", {})
    checkpoint_path = models_cfg.get("sidtd_checkpoint")
    if checkpoint_path:
        # Resolve relative to module root if needed
        ckpt = Path(checkpoint_path)
        if not ckpt.is_absolute():
            base_dir = Path(__file__).resolve().parent.parent.parent
            checkpoint_path = str(base_dir / checkpoint_path)

    detector_model = SIDTDEfficientNetDetector(
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes=models_cfg.get("num_classes", 2),
    )

    tamper_detector = TamperDetector(
        detector=detector_model,
        input_size=models_cfg.get("input_size", 300),
        tamper_threshold=models_cfg.get("tamper_threshold", 0.50),
        device=device,
    )

    # 2. Guilloché Reference Manager & Detector
    guilloche_cfg = cfg.get("guilloche", {})
    ref_root = guilloche_cfg.get("reference_root")
    if ref_root and not Path(ref_root).is_absolute():
        base_dir = Path(__file__).resolve().parent.parent.parent
        ref_root = str(base_dir / ref_root)

    reference_manager = GuillocheReferenceManager(reference_root=ref_root)
    guilloche_detector = GuillocheDetector(
        reference_manager=reference_manager,
        similarity_threshold=guilloche_cfg.get("similarity_threshold", 0.75),
    )

    # 3. Forensic Preprocessor
    ela_cfg = cfg.get("ela", {})
    dct_cfg = cfg.get("dct", {})
    forensic_preprocessor = ForensicPreprocessor(
        ela_quality=ela_cfg.get("quality", 90),
        ela_scale=ela_cfg.get("scale", 10),
        dct_block_size=dct_cfg.get("block_size", 8),
    )

    # 4. AI Gate Detector
    ai_cfg = cfg.get("ai_gate", {})
    ai_detector = ModularAIDetector(model_path=ai_cfg.get("model_path"))

    # 5. Generic Document Processor
    document_processor = DocumentProcessor(
        ai_detector=ai_detector,
        tamper_detector=tamper_detector,
        guilloche_detector=guilloche_detector,
        forensic_preprocessor=forensic_preprocessor,
        ai_input_size=ai_cfg.get("input_size", 224),
        device=device,
    )

    # 6. Scorer
    forensics_cfg = cfg.get("forensics", {})
    scorer = ForensicScorer(
        tamper_weight=forensics_cfg.get("tamper_weight", 0.60),
        guilloche_weight=forensics_cfg.get("guilloche_weight", 0.40),
        missing_ref_policy=forensics_cfg.get("scoring_policy_missing_ref", "REWEIGHT"),
    )

    max_workers = cfg.get("runtime", {}).get("max_workers", 4)

    return Module2Orchestrator(
        document_processor=document_processor,
        scorer=scorer,
        max_workers=max_workers,
    )


def run_module2(
    case_data: Union[Dict[str, Any], CaseInput],
    config_path: Optional[str] = None,
    orchestrator: Optional[Module2Orchestrator] = None,
) -> Dict[str, Any]:
    """Execute Module 2 on a case.

    Args:
        case_data: Dictionary or CaseInput instance.
        config_path: Path to configuration YAML.
        orchestrator: Optional cached orchestrator instance.

    Returns:
        Dictionary matching Risk Engine output contract.
    """
    if orchestrator is None:
        cfg = load_config(config_path)
        orchestrator = build_orchestrator(cfg)

    if isinstance(case_data, dict):
        case_input = CaseInput(
            uuid=case_data.get("uuid", ""),
            documents_present=case_data.get("documents_present", []),
            passport_image=case_data.get("passport_image"),
            visa_image=case_data.get("visa_image"),
            national_id_image=case_data.get("national_id_image"),
            driving_license_image=case_data.get("driving_license_image"),
            permit_image=case_data.get("permit_image"),
            metadata=case_data.get("metadata", {}),
        )
    else:
        case_input = case_data

    output: CaseOutput = orchestrator.process_case(case_input)
    return output.to_dict()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Module 2 — Visual / Image Forensics")
    parser.add_argument("--config", type=str, default=None, help="Path to module2_config.yaml")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        case_dict = json.load(f)

    result = run_module2(case_dict, config_path=args.config)
    output_json = json.dumps(result, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        logger.info(f"Results written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()
