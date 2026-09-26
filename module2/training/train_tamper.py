"""GPU training for PRAMAN Tamper / Manipulation Forensics.

Training input:
    dataset image -> RGB + ELA + DCT -> 5-channel tensor -> EfficientNet-B3

There is deliberately NO production upload-quality gate here. The production
gate belongs before Module 2 receives a queue item. Training uses the dataset
images directly.

One model is trained with:
1. binary forgery head: BONA_FIDE vs FORGED
2. manipulation-reason head: multi-label reasons supported by dataset metadata
"""

import argparse
import csv
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from module2.models.efficientnet_detector import (
    build_efficientnet_b3,
    CLASS_NAMES,
    MANIPULATION_CLASSES,
)
from module2.preprocessing.ela import ErrorLevelAnalysis
from module2.preprocessing.dct import DiscreteCosineTransform
from module2.preprocessing.raw_preprocessor import RawDocumentPreprocessor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [Tamper-Train] %(message)s")
logger = logging.getLogger("tamper_train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def map_field(field: str) -> str:
    f = (field or "").lower().replace("-", "_").replace(" ", "_")
    if any(x in f for x in ["name", "surname", "given"]):
        return "NAME_FIELD_MANIPULATION"
    if any(x in f for x in ["birth", "dob"]):
        return "DOB_FIELD_MANIPULATION"
    if any(x in f for x in ["passport", "document_number", "id_number", "number"]):
        return "DOCUMENT_NUMBER_FIELD_MANIPULATION"
    if any(x in f for x in ["photo", "portrait", "face"]):
        return "PHOTO_REPLACEMENT"
    return "GENERAL_MANIPULATION"


def map_ctype(ctype: str) -> str:
    c = (ctype or "").lower().replace("-", "_").replace(" ", "_")
    if any(x in c for x in ["crop", "replace", "copy", "paste", "move", "splice"]):
        return "COPY_PASTE_SPLICING"
    if any(x in c for x in ["inpaint", "erase", "remove", "rewrite"]):
        return "ERASURE_REWRITING"
    return "GENERAL_MANIPULATION"


def reason_vector(reasons: List[str]) -> np.ndarray:
    out = np.zeros(len(MANIPULATION_CLASSES), dtype=np.float32)
    for r in reasons:
        if r in MANIPULATION_CLASSES:
            out[MANIPULATION_CLASSES.index(r)] = 1.0
    return out


def resolve_path(root: Path, raw: str) -> Path | None:
    if not raw:
        return None
    clean = raw.replace("\\", "/").strip()
    candidates = [
        root / clean,
        root / "templates" / clean,
        root / "templates" / "Images" / clean,
        root / "Images" / clean,
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    return None


def find_sidtd_annotation(root: Path, image_path: Path) -> Path | None:
    stem = image_path.stem
    for p in [
        root / "templates" / "Annotations" / "fakes" / f"{stem}.json",
        root / "Annotations" / "fakes" / f"{stem}.json",
    ]:
        if p.is_file():
            return p
    matches = list(root.glob(f"**/Annotations/**/fakes/{stem}.json"))
    return matches[0] if matches else None


def sidtd_reasons(root: Path, image_path: Path) -> List[str]:
    ann = find_sidtd_annotation(root, image_path)
    if ann is None:
        return ["GENERAL_MANIPULATION"]
    try:
        data = json.loads(ann.read_text(encoding="utf-8"))
        reasons = {"GENERAL_MANIPULATION", map_ctype(str(data.get("ctype", "")))}
        field = str(data.get("field", ""))
        second_field = str(data.get("second_field", ""))
        reasons.add(map_field(field))
        if second_field:
            reasons.add(map_field(second_field))
        return [r for r in reasons if r in MANIPULATION_CLASSES]
    except Exception:
        return ["GENERAL_MANIPULATION"]


class TamperSample:
    def __init__(self, path: Path, label: int, reasons: List[str]):
        self.path, self.label, self.reasons = path, label, reasons


class TamperDataset(Dataset):
    def __init__(self, samples: List[TamperSample], image_size: int, train: bool):
        self.samples = samples
        self.size = image_size
        self.train = train
        self.ela = ErrorLevelAnalysis(default_quality=90, default_scale=10)
        self.dct = DiscreteCosineTransform(block_size=8)
        self.spatial = transforms.RandomHorizontalFlip(p=0.5) if train else transforms.Lambda(lambda x: x)
        self.rgb_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.raw_preprocessor = RawDocumentPreprocessor()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")

        # Training follows the same common document normalization used in production:
        # EXIF orientation -> boundary detection/cropping -> perspective correction -> CLAHE.
        # No upload-quality gate is applied to dataset images.
        img, _ = self.raw_preprocessor.preprocess_raw_document(img, doc_type="training_document")

        # One shared spatial transform is applied before deriving ELA/DCT.
        if self.train and random.random() < 0.5:
            img = transforms.functional.hflip(img)

        img = img.resize((self.size, self.size), Image.Resampling.BILINEAR)

        rgb = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        rgb = self.rgb_norm(rgb)

        ela = self.ela.compute_ela(img, quality=90, scale=10).convert("L")
        ela_t = torch.from_numpy(np.asarray(ela, dtype=np.float32) / 255.0).unsqueeze(0)

        dct = self.dct.compute_dct_map(img)
        dct = np.log1p(np.abs(dct))
        dct = (dct - dct.min()) / max(float(dct.max() - dct.min()), 1e-6)
        dct_t = torch.from_numpy(dct.astype(np.float32)).unsqueeze(0)

        x = torch.cat([rgb, ela_t, dct_t], dim=0)
        y_binary = torch.tensor(s.label, dtype=torch.long)
        y_reason = torch.tensor(reason_vector(s.reasons), dtype=torch.float32)
        return x, y_binary, y_reason


def load_sidtd(root: Path, split: str) -> List[TamperSample]:
    csv_path = root / "split_normal" / f"{split}_split_SIDTD.csv"
    if not csv_path.exists():
        return []
    out = []
    with csv_path.open("r", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            path = resolve_path(root, row.get("image_path", ""))
            if not path:
                continue
            label = int(row.get("label", "0"))
            reasons = sidtd_reasons(root, path) if label == 1 else []
            out.append(TamperSample(path, label, reasons))
    return out


def load_casia(root: Path, split: str, seed: int) -> List[TamperSample]:
    samples = []
    for p in (root / "Au").glob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            samples.append(TamperSample(p, 0, []))
    for p in (root / "Tp").glob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            # CASIA supports general manipulation supervision here; do not
            # invent field-specific labels from filenames.
            samples.append(TamperSample(p, 1, ["GENERAL_MANIPULATION"]))

    rng = random.Random(seed)
    rng.shuffle(samples)
    cut = int(len(samples) * 0.8)
    return samples[:cut] if split == "train" else samples[cut:]


def load_midv(root: Path, split: str, seed: int) -> List[TamperSample]:
    candidates = list(root.glob("**/images/**/*"))
    images = [p for p in candidates if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    rng = random.Random(seed)
    rng.shuffle(images)
    cut = int(len(images) * 0.8)
    selected = images[:cut] if split == "train" else images[cut:]
    return [TamperSample(p, 0, []) for p in selected]


def load_samples(cfg: dict, split: str) -> List[TamperSample]:
    root = Path(cfg["data"]["dataset_root"])
    out = []
    ds = cfg["data"].get("dataset_type", "combined")

    if ds in ("sidtd", "combined"):
        out.extend(load_sidtd(root, split))
    if ds in ("casia2", "combined"):
        out.extend(load_casia(root / "CASIA-v2.0", split, cfg["runtime"]["seed"]))
    if ds in ("midv2020", "combined"):
        out.extend(load_midv(root / "MIDV-2020", split, cfg["runtime"]["seed"]))

    random.shuffle(out)
    return out


def evaluate(model, loader, device, criterion_binary, criterion_reason):
    model.eval()
    losses = []
    preds, targets, probs = [], [], []
    reason_true, reason_pred = [], []

    with torch.no_grad():
        for x, y, yr in loader:
            x, y, yr = x.to(device), y.to(device), yr.to(device)
            out = model(x)
            loss_b = criterion_binary(out["forgery_logits"], y)
            forged_mask = y == 1
            loss_r = criterion_reason(out["reason_logits"][forged_mask], yr[forged_mask]) if forged_mask.any() else torch.tensor(0.0, device=device)
            losses.append(float((loss_b + loss_r).item()))

            p = torch.softmax(out["forgery_logits"], 1)[:, 1]
            preds.extend((p >= 0.5).long().cpu().tolist())
            targets.extend(y.cpu().tolist())
            probs.extend(p.cpu().tolist())
            if forged_mask.any():
                reason_true.extend(yr[forged_mask].cpu().numpy().tolist())
                reason_pred.extend((torch.sigmoid(out["reason_logits"][forged_mask]) >= 0.5).cpu().numpy().tolist())

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy_score(targets, preds) if targets else 0.0,
        "precision": precision_score(targets, preds, zero_division=0) if targets else 0.0,
        "recall": recall_score(targets, preds, zero_division=0) if targets else 0.0,
        "f1": f1_score(targets, preds, zero_division=0) if targets else 0.0,
        "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1]).tolist() if targets else [[0, 0], [0, 0]],
    }
    metrics["roc_auc"] = roc_auc_score(targets, probs) if len(set(targets)) > 1 else None
    if reason_true:
        rt, rp = np.array(reason_true), np.array(reason_pred)
        metrics["reason_f1_micro"] = float(f1_score(rt, rp, average="micro", zero_division=0))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/tamper_training_config.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["runtime"]["seed"])

    device_name = args.device or cfg["runtime"].get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    logger.info("Training device: %s", device)

    train_samples = load_samples(cfg, "train")
    val_samples = load_samples(cfg, "val")
    if args.dry_run:
        train_samples, val_samples = train_samples[:64], val_samples[:32]

    logger.info("Samples: train=%d val=%d", len(train_samples), len(val_samples))
    train_ds = TamperDataset(train_samples, cfg["training"]["image_size"], True)
    val_ds = TamperDataset(val_samples, cfg["training"]["image_size"], False)

    loader_args = dict(
        batch_size=args.batch_size or cfg["training"]["batch_size"],
        num_workers=cfg["runtime"]["num_workers"],
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    model = build_efficientnet_b3(
        num_classes=2,
        reason_classes=len(MANIPULATION_CLASSES),
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        input_channels=5,
    ).to(device)

    criterion_binary = nn.CrossEntropyLoss()
    criterion_reason = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs or cfg["training"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["runtime"]["mixed_precision"] and device.type == "cuda")

    best_f1 = -1.0
    history = []
    epochs = 1 if args.dry_run else (args.epochs or cfg["training"]["epochs"])
    output_dir = Path(cfg["output"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running = []
        for x, y, yr in train_loader:
            x, y, yr = x.to(device), y.to(device), yr.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x)
                loss_b = criterion_binary(out["forgery_logits"], y)
                forged_mask = y == 1
                loss_r = criterion_reason(out["reason_logits"][forged_mask], yr[forged_mask]) if forged_mask.any() else torch.tensor(0.0, device=device)
                loss = loss_b + cfg["training"]["reason_loss_weight"] * loss_r
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running.append(float(loss.item()))

        scheduler.step()
        metrics = evaluate(model, val_loader, device, criterion_binary, criterion_reason)
        metrics["train_loss"] = float(np.mean(running)) if running else 0.0
        metrics["epoch"] = epoch
        history.append(metrics)
        logger.info("Epoch %d/%d | train_loss=%.4f | val_f1=%.4f | val_auc=%s", epoch, epochs, metrics["train_loss"], metrics["f1"], metrics["roc_auc"])

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "model_name": "efficientnet_b3",
                "input_channels": 5,
                "forensic_inputs": ["RGB", "ELA", "DCT"],
                "class_names": CLASS_NAMES,
                "reason_classes": MANIPULATION_CLASSES,
                "reason_threshold": 0.50,
                "tamper_threshold": 0.50,
                "val_metrics": metrics,
                "training_config": cfg,
            }
            torch.save(checkpoint, output_dir / cfg["output"]["checkpoint_filename"])
            (output_dir / "tamper_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    logger.info("Best checkpoint: %s", output_dir / cfg["output"]["checkpoint_filename"])


if __name__ == "__main__":
    main()
