"""Unified training script for Splice/Tamper Detection in Module 2.2.

Supports:
1. Datasets:
   - CASIA v2.0 (image splicing and copy-move tampering)
   - SIDTD (identity document forgery benchmark, split_normal)
   - MIDV-2020 (identity document tampering adaptation)
   - Combined (progressive multi-dataset training)
2. Models:
   - EfficientNet-B3
   - ResNet-18
3. Multi-modal forensic feature representations (optional ELA context).
4. HPC execution (CUDA, MPS, AMP mixed precision, Cosine Annealing, experiment logging).
"""

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import (
    EfficientNet_B3_Weights,
    ResNet18_Weights,
    efficientnet_b3,
    resnet18,
)
from PIL import Image, ImageOps
import yaml
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Ensure src/ is in Python path for RawDocumentPreprocessor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from module2.preprocessing.raw_preprocessor import RawDocumentPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Tamper-Train]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("tamper_train")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TamperDataset(Dataset):
    """Five-channel forensic dataset: RGB + ELA + DCT."""
    def __init__(self, samples, transform=None, enable_raw_preprocessing=True):
        self.samples=samples; self.transform=transform; self.enable_raw_preprocessing=enable_raw_preprocessing
        self.preprocessor=RawDocumentPreprocessor() if enable_raw_preprocessing else None
        from module2.preprocessing.ela import ErrorLevelAnalysis
        from module2.preprocessing.dct import DiscreteCosineTransform
        self.ela=ErrorLevelAnalysis(default_quality=90, default_scale=10)
        self.dct=DiscreteCosineTransform(block_size=8)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path,label=self.samples[idx]
        try:
            with open(img_path,"rb") as f: img=Image.open(f).convert("RGB")
            if self.preprocessor is not None:
                try: img,_=self.preprocessor.preprocess_raw_document(img)
                except Exception as e: logger.debug(f"Raw preprocessing fallback for {img_path}: {e}")
            ela=self.ela.compute_ela(img,quality=90,scale=10).convert("L")
            dct=self.dct.compute_dct_map(img)
            dct=np.log1p(np.abs(dct)); dct=(dct-dct.min())/max(float(dct.max()-dct.min()),1e-6)
            dct_img=Image.fromarray((dct*255).astype(np.uint8),"L")
            rgb=self.transform(img) if self.transform else transforms.ToTensor()(img)
            ela_t=(self.transform(ela.convert("RGB")) if self.transform else transforms.ToTensor()(ela))[0:1]
            dct_t=(self.transform(dct_img.convert("RGB")) if self.transform else transforms.ToTensor()(dct_img))[0:1]
            return torch.cat([rgb,ela_t,dct_t],dim=0),label
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            return torch.zeros(5,300,300),label


def get_transforms(image_size: int = 300) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # Spatial augmentation is applied once in TamperDataset so RGB/ELA/DCT remain aligned.
    # Do not apply independent random transforms to the three modalities.
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def build_model(
    model_name: str = "efficientnet_b3",
    num_classes: int = 2,
    pretrained: bool = False,
    dropout: float = 0.3,
) -> nn.Module:
    """Build classifier backbone (EfficientNet-B3 or ResNet-18)."""
    name = model_name.lower().strip()
    if name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        return model
    else:
        # Default: EfficientNet-B3
        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        return model


def resolve_image_path(raw_path: str, dataset_root: Path) -> Optional[Path]:
    """Resolve image path to an existing local file."""
    if not raw_path:
        return None

    clean = str(raw_path).replace("\\", "/").strip()
    p = Path(clean)
    if p.is_absolute() and p.is_file():
        return p

    candidates = [
        dataset_root / clean,
        dataset_root / "templates" / clean,
        dataset_root / "templates" / "Images" / clean,
        dataset_root / "Images" / clean,
        dataset_root.parent / clean,
        Path.cwd() / clean,
    ]

    if clean.startswith("templates/"):
        sub = clean[len("templates/"):]
        candidates.append(dataset_root / "templates" / sub)
        candidates.append(dataset_root / sub)
        candidates.append(dataset_root / "templates" / "Images" / sub)

    if clean.startswith("Images/"):
        sub = clean[len("Images/"):]
        candidates.append(dataset_root / "templates" / clean)
        candidates.append(dataset_root / clean)
        candidates.append(dataset_root / "templates" / "Images" / sub)

    for cand in candidates:
        try:
            if cand.is_file():
                return cand.resolve()
        except OSError:
            continue

    return None


def parse_sidtd_csv(csv_path: Path, dataset_root: Path) -> List[Tuple[str, int]]:
    """Parse SIDTD split CSV into (image_path, label) pairs."""
    samples: List[Tuple[str, int]] = []
    if not csv_path.exists():
        return samples

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None and v is not None}
            label = None
            if "label" in clean_row and clean_row["label"] != "":
                try:
                    label = int(clean_row["label"])
                except ValueError:
                    pass

            if label is None and "label_name" in clean_row:
                lname = clean_row["label_name"].lower()
                if any(x in lname for x in ("bona", "real", "0")):
                    label = 0
                elif any(x in lname for x in ("forged", "fake", "tamper", "1")):
                    label = 1

            if label is None:
                continue

            raw_path = clean_row.get("image_path") or clean_row.get("path") or clean_row.get("image")
            if not raw_path:
                continue

            resolved = resolve_image_path(raw_path, dataset_root)
            if resolved:
                samples.append((str(resolved), label))

    return samples


def load_casia2_dataset(casia_root: Path) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Parse CASIA v2.0 dataset (Au = Authentic / 0, Tp = Tampered / 1)."""
    samples: List[Tuple[str, int]] = []
    au_dir = casia_root / "Au"
    tp_dir = casia_root / "Tp"

    if au_dir.exists() and au_dir.is_dir():
        for ext in ("*.jpg", "*.tif", "*.png", "*.bmp"):
            for p in au_dir.glob(ext):
                samples.append((str(p), 0))

    if tp_dir.exists() and tp_dir.is_dir():
        for ext in ("*.jpg", "*.tif", "*.png", "*.bmp"):
            for p in tp_dir.glob(ext):
                samples.append((str(p), 1))

    if not samples:
        logger.warning(f"No CASIA v2.0 images found in {casia_root}")
        return [], []

    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    return samples[:split_idx], samples[split_idx:]


def _collect_midv_authentic_images(midv_root: Path) -> List[Path]:
    """Collect authentic MIDV-2020 document images.

    MIDV-2020 is a bona-fide identity-document dataset; it does not provide
    forged/tampered image labels. Therefore MIDV samples are intentionally
    added only as class-0 (BONA_FIDE) samples. Forged supervision comes from
    CASIA v2.0 and SIDTD.
    """
    candidates = [
        midv_root / "templates" / "images",
        midv_root / "dataset" / "templates" / "images",
        midv_root / "dataset" / "images",
    ]
    images: List[Path] = []
    seen = set()
    for root in candidates:
        if not root.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
            for p in root.rglob(ext):
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    images.append(p)
    return images


def load_midv_dataset(midv_root: Path) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Load MIDV-2020 as bona-fide-only training data.

    MIDV-2020 contains authentic mock documents and rich annotations, but is
    not a forged-vs-real dataset. We therefore never infer a forged label
    from a filename. This adapter contributes only label 0 samples.
    """
    images = _collect_midv_authentic_images(midv_root)
    if not images:
        logger.warning(f"No MIDV-2020 images found in {midv_root}")
        return [], []

    samples = [(str(p), 0) for p in images]
    random.shuffle(samples)
    split_idx = max(1, int(len(samples) * 0.8))
    return samples[:split_idx], samples[split_idx:]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = False,
) -> float:
    """Train classifier for one epoch."""
    model.train()
    running_loss = 0.0
    steps = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        steps += 1

    return running_loss / max(1, steps)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate classifier and compute full metrics."""
    model.eval()
    running_loss = 0.0
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

    n_batches = max(1, len(loader))
    val_loss = running_loss / n_batches

    if not all_targets:
        return {"loss": val_loss, "accuracy": 0.0}

    accuracy = sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
    except Exception:
        auc = 0.5

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1]).tolist()

    return {
        "loss": float(val_loss),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "confusion_matrix": cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Splice/Tamper Detection Model")
    parser.add_argument("--config", type=str, default="training/tamper_training_config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset-type", type=str, default=None, choices=["sidtd", "casia2", "midv2020", "combined"])
    parser.add_argument("--model-name", type=str, default=None, choices=["efficientnet_b3", "resnet18"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None, help="Compute device override (cuda, mps, cpu)")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint filename")
    parser.add_argument("--resume", type=str, default=None, help="Resume/fine-tune from existing checkpoint")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable raw document preprocessing during training")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 quick epoch to verify pipeline")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent / "tamper_training_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    set_seed(cfg.get("runtime", {}).get("seed", 42))

    # Resolve device
    dev_str = args.device or cfg.get("runtime", {}).get("device", "auto")
    if dev_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(dev_str)

    logger.info(f"Target compute device: {device}")

    # Dataset loading
    ds_type = (args.dataset_type or cfg.get("data", {}).get("dataset_type", "sidtd")).lower()
    dataset_root = Path(cfg.get("data", {}).get("dataset_root", "data"))
    if not dataset_root.is_absolute():
        dataset_root = Path(__file__).resolve().parent.parent / dataset_root

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []

    if ds_type in ("sidtd", "combined"):
        split_dir = dataset_root / cfg.get("data", {}).get("split_type", "split_normal")
        train_csv = split_dir / "train_split_SIDTD.csv"
        val_csv = split_dir / "val_split_SIDTD.csv"
        sidtd_train = parse_sidtd_csv(train_csv, dataset_root)
        sidtd_val = parse_sidtd_csv(val_csv, dataset_root)
        logger.info(f"Loaded SIDTD: {len(sidtd_train)} train, {len(sidtd_val)} val samples.")
        train_samples.extend(sidtd_train)
        val_samples.extend(sidtd_val)

    if ds_type in ("casia2", "combined"):
        casia_dir = Path(cfg.get("data", {}).get("casia_root", "data/CASIA-v2.0"))
        if not casia_dir.is_absolute():
            casia_dir = Path(__file__).resolve().parent.parent / casia_dir
        c_train, c_val = load_casia2_dataset(casia_dir)
        if c_train:
            logger.info(f"Loaded CASIA v2.0: {len(c_train)} train, {len(c_val)} val samples.")
            train_samples.extend(c_train)
            val_samples.extend(c_val)

    if ds_type in ("midv2020", "combined"):
        midv_dir = Path(cfg.get("data", {}).get("midv_root", "data/MIDV-2020"))
        if not midv_dir.is_absolute():
            midv_dir = Path(__file__).resolve().parent.parent / midv_dir
        m_train, m_val = load_midv_dataset(midv_dir)
        if m_train:
            logger.info(f"Loaded MIDV-2020: {len(m_train)} train, {len(m_val)} val samples.")
            train_samples.extend(m_train)
            val_samples.extend(m_val)

    if not train_samples:
        logger.error(f"No training samples found for dataset type '{ds_type}'. Exiting.")
        sys.exit(1)

    if args.dry_run:
        train_samples = train_samples[:32]
        val_samples = val_samples[:16]

    logger.info(f"Dataset summary ({ds_type}): {len(train_samples)} train, {len(val_samples)} val.")

    img_size = cfg.get("training", {}).get("image_size", 300)
    train_tf, val_tf = get_transforms(image_size=img_size)

    enable_raw_prep = not args.no_preprocess and cfg.get("preprocessing", {}).get("enabled", True)
    logger.info(f"Raw Document Preprocessing for training: {'ENABLED' if enable_raw_prep else 'DISABLED'}")

    train_ds = TamperDataset(train_samples, transform=train_tf, enable_raw_preprocessing=enable_raw_prep)
    val_ds = TamperDataset(val_samples, transform=val_tf, enable_raw_preprocessing=enable_raw_prep)

    batch_size = 4 if args.dry_run else (args.batch_size or cfg.get("training", {}).get("batch_size", 16))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model_name = args.model_name or cfg.get("model", {}).get("name", "efficientnet_b3")
    pretrained = cfg.get("model", {}).get("pretrained", False)
    dropout = cfg.get("model", {}).get("dropout", 0.3)
    num_classes = cfg.get("model", {}).get("num_classes", 2)

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    if model_name.lower() == "efficientnet_b3":
        from module2.models.efficientnet_detector import build_efficientnet_b3
        model = build_efficientnet_b3(num_classes=num_classes, pretrained=pretrained, dropout=dropout, input_channels=5)

    # Resume / Fine-tune from checkpoint if provided
    resume_path = args.resume or cfg.get("training", {}).get("resume")
    if resume_path:
        rp = Path(resume_path)
        if rp.exists():
            try:
                ckpt = torch.load(rp, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
                model.load_state_dict(state, strict=False)
                logger.info(f"Loaded starting weights from checkpoint: {rp}")
            except Exception as e:
                logger.warning(f"Failed to load resume checkpoint from {rp}: {e}")

    model.to(device)

    # Loss & Optimizer
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 0.0003)
    weight_decay = cfg.get("training", {}).get("weight_decay", 0.0001)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 1 if args.dry_run else (args.epochs or cfg.get("training", {}).get("epochs", 25))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = cfg.get("runtime", {}).get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Output directory
    out_dir = Path(__file__).resolve().parent.parent / cfg.get("output", {}).get("checkpoint_dir", "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = args.output or cfg.get("output", {}).get("checkpoint_filename", "efficientnet_b3_combined_tamper.pth")
    if args.dry_run and not args.output:
        out_filename = "dryrun_tamper.pth"
    out_path = out_dir / Path(out_filename).name

    best_f1 = 0.0
    best_loss = float("inf")
    patience = cfg.get("training", {}).get("patience", 5)
    patience_counter = 0

    logger.info(f"Starting Splice/Tamper training ({model_name}) for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp)
        scheduler.step()

        val_metrics = evaluate(model, val_loader, criterion, device)
        v_loss = val_metrics["loss"]
        v_acc = val_metrics["accuracy"]
        v_f1 = val_metrics.get("f1", 0.0)
        v_auc = val_metrics.get("auc", 0.5)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} | "
            f"Val Acc: {v_acc:.4f} | "
            f"Val F1: {v_f1:.4f} | "
            f"Val AUC: {v_auc:.4f}"
        )

        is_best = v_f1 > best_f1 or (v_f1 == best_f1 and v_loss < best_loss)
        if is_best:
            best_f1 = v_f1
            best_loss = v_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_name": model_name,
                "class_names": cfg.get("model", {}).get("class_names", ["BONA_FIDE", "FORGED"]),
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "training_config": cfg,
            }
            if cfg.get("output", {}).get("save_optimizer", False):
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()

            torch.save(checkpoint, out_path)
            logger.info(f"[*] Best model checkpoint saved to: {out_path} (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    logger.info(f"Tamper training complete. Best checkpoint saved at: {out_path}")


if __name__ == "__main__":
    main()
