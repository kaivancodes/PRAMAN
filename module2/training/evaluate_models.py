"""Held-out evaluation for the two Module 2 forensic models.

No production upload-quality gate is used here.
Evaluation starts with dataset images and applies the same branch-specific
deterministic preprocessing used by the trained models.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from module2.models.efficientnet_detector import SIDTDEfficientNetDetector
from module2.models.siamese_guilloche import SiameseResNet18
from training.train_tamper import TamperDataset, load_samples


def evaluate_tamper(cfg, checkpoint, device):
    samples = load_samples(cfg, "val")
    if not samples:
        raise RuntimeError("No held-out Tamper samples found.")

    ds = TamperDataset(samples, cfg["training"]["image_size"], False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    detector = SIDTDEfficientNetDetector(str(checkpoint), device=device)
    detector.model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.inference_mode():
        for x, y, _ in loader:
            out = detector.model(x.to(device))
            p = torch.softmax(out["forgery_logits"], dim=1)[:, 1]
            y_true.extend(y.tolist())
            y_pred.extend((p >= 0.5).long().cpu().tolist())
            y_prob.extend(p.cpu().tolist())

    print("\nTAMPER / MANIPULATION FORENSICS")
    print(f"Samples    : {len(y_true)}")
    print(f"Accuracy   : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision  : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall     : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1         : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if len(set(y_true)) > 1:
        print(f"ROC-AUC    : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Confusion  : {confusion_matrix(y_true, y_pred, labels=[0,1]).tolist()}")


def main():
    import yaml

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/tamper_training_config.yaml")
    ap.add_argument("--checkpoint", default="models/efficientnet_b3_combined_tamper.pth")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto"
        else "cpu"
    )
    evaluate_tamper(cfg, Path(args.checkpoint), device)


if __name__ == "__main__":
    main()
