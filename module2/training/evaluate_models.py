"""Held-out evaluation for Module 2.2 models.

This is model evaluation, not unit/integration testing. It applies the same raw
document preprocessing used by training/inference before the final tensor
transforms, and reports classification/metric-learning accuracy metrics.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from module2.models.siamese_guilloche import SiameseResNet18
from module2.preprocessing.raw_preprocessor import RawDocumentPreprocessor
from module2.models.efficientnet_detector import SIDTDEfficientNetDetector
from training.train_tamper import TamperDataset, get_transforms, parse_sidtd_csv


def resolve(root: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    candidates = [root / raw, root / "templates" / raw, root / "templates" / "Images" / raw]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(raw)


def evaluate_tamper(test_csv: Path, dataset_root: Path, checkpoint: Path, device: torch.device) -> None:
    samples = parse_sidtd_csv(test_csv, dataset_root)
    if not samples:
        raise RuntimeError(f"No held-out SIDTD test samples found in {test_csv}")

    _, test_tf = get_transforms(300)
    ds = TamperDataset(samples, transform=test_tf, enable_raw_preprocessing=True)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = SIDTDEfficientNetDetector(checkpoint_path=str(checkpoint), device=device)
    model.model.eval()

    y_true, y_pred, y_prob = [], [], []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            logits = model.model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

    print("\nTAMPER MODEL — HELD-OUT SIDTD TEST")
    print(f"Samples    : {len(y_true)}")
    print(f"Accuracy   : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision  : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall     : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1         : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if len(set(y_true)) > 1:
        print(f"ROC-AUC    : {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Confusion  : {confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()}")


def make_pairs(real: List[str], fake: List[str], count: int) -> List[Tuple[str, str, int]]:
    import random
    rng = random.Random(1234)
    if len(real) < 2:
        raise RuntimeError("Need at least two authentic images for Guilloché evaluation.")
    pairs = []
    for _ in range(count // 2):
        a, b = rng.sample(real, 2)
        pairs.append((a, b, 1))
    for _ in range(count - len(pairs)):
        if fake:
            pairs.append((rng.choice(real), rng.choice(fake), -1))
        else:
            a, b = rng.sample(real, 2)
            pairs.append((a, b, -1))
    return pairs


def evaluate_guilloche(dataset_root: Path, checkpoint: Path, device: torch.device, threshold: float = 0.75) -> None:
    real_dir = dataset_root / "templates" / "Images" / "reals"
    fake_dir = dataset_root / "templates" / "Images" / "fakes"
    real = [str(p) for p in real_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    fake = [str(p) for p in fake_dir.rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    if len(real) < 2:
        raise RuntimeError("No sufficient SIDTD Guilloché evaluation images found.")

    pairs = make_pairs(real, fake, min(1000, max(100, len(real))))
    prep = RawDocumentPreprocessor()
    model = SiameseResNet18(embedding_dim=128, pretrained=False)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)), strict=False)
    model.to(device).eval()

    sims, labels = [], []
    with torch.inference_mode():
        for p1, p2, label in pairs:
            i1 = prep.preprocess_raw_document(Image.open(p1).convert("RGB"))[0].resize((128,128))
            i2 = prep.preprocess_raw_document(Image.open(p2).convert("RGB"))[0].resize((128,128))
            import torchvision.transforms as T
            tf = T.Compose([T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
            e1 = model.extract_features(tf(i1).unsqueeze(0).to(device))
            e2 = model.extract_features(tf(i2).unsqueeze(0).to(device))
            sims.append(float(F.cosine_similarity(e1,e2).item()))
            labels.append(1 if label > 0 else 0)

    preds = [1 if s >= threshold else 0 for s in sims]
    print("\nGUILLOCHÉ MODEL — HELD-OUT EVALUATION")
    print(f"Pairs      : {len(labels)}")
    print(f"Threshold  : {threshold:.2f}")
    print(f"Accuracy   : {accuracy_score(labels, preds):.4f}")
    print(f"ROC-AUC    : {roc_auc_score(labels, sims):.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", default="data")
    p.add_argument("--tamper-checkpoint", default="models/efficientnet_b3_combined_tamper.pth")
    p.add_argument("--guilloche-checkpoint", default="models/guilloche_siamese_resnet18.pth")
    p.add_argument("--device", default="auto")
    p.add_argument("--skip-tamper", action="store_true")
    p.add_argument("--skip-guilloche", action="store_true")
    p.add_argument("--guilloche-threshold", type=float, default=0.75)
    a = p.parse_args()

    device = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else a.device if a.device != "auto" else "cpu")
    root = Path(a.dataset_root)
    if not root.is_absolute():
        root = Path(__file__).resolve().parent.parent / root

    if not a.skip_tamper:
        evaluate_tamper(root / "split_normal" / "test_split_SIDTD.csv", root, Path(a.tamper_checkpoint), device)
    if not a.skip_guilloche:
        evaluate_guilloche(root, Path(a.guilloche_checkpoint), device, a.guilloche_threshold)


if __name__ == "__main__":
    main()
