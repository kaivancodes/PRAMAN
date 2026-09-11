"""Training script for SIDTD Document Forgery Detection using EfficientNet-B3.

Trains binary classifier: BONA_FIDE (0) vs FORGED (1) using predefined SIDTD split_normal.
CRITICAL: SIDTD is strictly for document forgery/tamper detection, NOT AI-generation.
This script is completely separated from production inference.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
import yaml
from PIL import Image

# Setup training logger
import logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [SIDTD-Train]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sidtd_train")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SIDTDDataset(Dataset):
    """Dataset loader for SIDTD predefined split (split_normal).

    Labels:
      0: BONA_FIDE
      1: FORGED
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        try:
            with open(img_path, "rb") as f:
                img = Image.open(f).convert("RGB")
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}. Returning zero tensor.")
            img = Image.new("RGB", (300, 300), color=0)

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(image_size: int = 300) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def build_model(num_classes: int = 2, pretrained: bool = True, dropout: float = 0.3) -> nn.Module:
    """Instantiate EfficientNet-B3 with binary classification head."""
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def resolve_image_path(raw_path: str, dataset_root: Path) -> Optional[Path]:
    """Resolve an image path from SIDTD CSV to an existing local file."""
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


def _parse_sidtd_csv(csv_path: Path, dataset_root: Path) -> List[Tuple[str, int]]:
    """Parse SIDTD split CSV into (image_path, label) pairs.

    Handles columns: label_name, label, image_path, class, class_name.
    Labels:
      0 = BONA_FIDE (reals)
      1 = FORGED (fakes)
    """
    samples: List[Tuple[str, int]] = []
    if not csv_path.exists():
        return samples

    missing_count = 0
    first_missing = None

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean_row = {k.strip(): v.strip() for k, v in row.items() if k is not None and v is not None}

            # Extract label
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
            else:
                missing_count += 1
                if first_missing is None:
                    first_missing = raw_path

    if missing_count > 0:
        logger.warning(
            f"From {csv_path.name}: {len(samples)} valid images found, {missing_count} missing from disk. "
            f"(Example missing path in CSV: {first_missing})"
        )
    else:
        logger.info(f"Loaded {len(samples)} samples from {csv_path.name}")

    return samples


def load_sidtd_split(
    dataset_root: Path,
    split_type: str = "split_normal",
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Parse SIDTD CSV split files or directory structure into train, val, test lists.

    Supports:
    1. Standard SIDTD CSV structure:
       dataset_root / [Split_normal|split_normal] /
         train_split_SIDTD.csv
         val_split_SIDTD.csv
         test_split_SIDTD.csv
    2. Fallback folder structure:
       dataset_root / split_type / [train|val|test] / [bona_fide|forged]
    """
    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []

    # 1. Look for split directory (case-tolerant)
    candidate_split_dirs = [
        dataset_root / split_type,
        dataset_root / "Split_normal",
        dataset_root / "split_normal",
        dataset_root,
    ]

    split_dir = None
    for cand in candidate_split_dirs:
        if cand.exists() and cand.is_dir():
            csvs = list(cand.glob("*.csv"))
            if csvs:
                split_dir = cand
                break

    # If not found yet, check any child dir with 'split' in name
    if split_dir is None and dataset_root.exists():
        for child in dataset_root.iterdir():
            if child.is_dir() and "split" in child.name.lower():
                if list(child.glob("*.csv")):
                    split_dir = child
                    break

    def _find_csv(folder: Path, split_name: str) -> Optional[Path]:
        patterns = [
            f"{split_name}_split_SIDTD.csv",
            f"{split_name}_split_sidtd.csv",
            f"{split_name}_split.csv",
            f"{split_name}.csv",
        ]
        for pat in patterns:
            target = folder / pat
            if target.exists():
                return target
        for f in folder.glob("*.csv"):
            if split_name.lower() in f.name.lower():
                return f
        return None

    if split_dir is not None:
        train_csv = _find_csv(split_dir, "train")
        val_csv = _find_csv(split_dir, "val")
        test_csv = _find_csv(split_dir, "test")

        if train_csv or val_csv or test_csv:
            logger.info(f"Detected SIDTD CSV split folder at: {split_dir}")
            if train_csv:
                train_samples = _parse_sidtd_csv(train_csv, dataset_root)
            if val_csv:
                val_samples = _parse_sidtd_csv(val_csv, dataset_root)
            if test_csv:
                test_samples = _parse_sidtd_csv(test_csv, dataset_root)
            return train_samples, val_samples, test_samples

    # Fallback to directory scan
    dir_target = dataset_root / split_type
    if not dir_target.exists():
        for cand in [dataset_root / "Split_normal", dataset_root / "split_normal", dataset_root]:
            if cand.exists():
                dir_target = cand
                break

    def _collect_from_dir(target_dir: Path) -> List[Tuple[str, int]]:
        items = []
        if not target_dir.exists():
            return items

        for entry in target_dir.iterdir():
            if entry.is_dir():
                name_lower = entry.name.lower()
                if "bona" in name_lower or "template" in name_lower or "real" in name_lower:
                    label = 0  # BONA_FIDE
                elif "forged" in name_lower or "fake" in name_lower or "tamper" in name_lower:
                    label = 1  # FORGED
                else:
                    continue

                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
                    for f in entry.glob(ext):
                        items.append((str(f), label))
        return items

    if dir_target.exists():
        train_samples = _collect_from_dir(dir_target / "train")
        val_samples = _collect_from_dir(dir_target / "val")
        test_samples = _collect_from_dir(dir_target / "test")

    return train_samples, val_samples, test_samples


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, float]:
    """Execute one training epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler and device.type == "cuda":
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = total_loss / max(1, total)
    epoch_acc = (correct / max(1, total)) * 100.0
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    val_loss = total_loss / max(1, total)
    val_acc = (correct / max(1, total)) * 100.0
    return val_loss, val_acc


def run_training(
    config_path: str,
    dataset_root_override: Optional[str] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    device_override: Optional[str] = None,
) -> None:
    """Execute complete SIDTD EfficientNet-B3 training workflow."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Command-line overrides
    if dataset_root_override:
        cfg.setdefault("data", {})["dataset_root"] = dataset_root_override
    if epochs_override:
        cfg.setdefault("training", {})["epochs"] = epochs_override
    if batch_size_override:
        cfg.setdefault("training", {})["batch_size"] = batch_size_override
    if device_override:
        cfg.setdefault("runtime", {})["device"] = device_override

    runtime_cfg = cfg.get("runtime", {})
    seed = runtime_cfg.get("seed", 42)
    set_seed(seed)

    # Determine device (with Apple Silicon MPS support)
    req_device = runtime_cfg.get("device", "auto")
    if req_device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(req_device)
    logger.info(f"Using training device: {device}")

    # Data
    data_cfg = cfg.get("data", {})
    dataset_root = Path(data_cfg.get("dataset_root", "data/sidtd"))
    split_type = data_cfg.get("split_type", "split_normal")

    logger.info(f"Checking dataset root: {dataset_root.resolve()}")
    train_samples, val_samples, _ = load_sidtd_split(dataset_root, split_type)
    if not train_samples:
        logger.error(
            f"No training samples found in {dataset_root}. "
            "Please ensure SIDTD dataset is placed with structure:\n"
            f"  {dataset_root}/\n"
            "    Split_normal/ (containing train_split_SIDTD.csv, val_split_SIDTD.csv, test_split_SIDTD.csv)\n"
            "    templates/Images/ (containing reals/ and fakes/ images)\n"
            "Or pass --dataset-root pointing to your local SIDTD directory."
        )
        return

    train_cfg = cfg.get("training", {})
    image_size = train_cfg.get("image_size", 300)
    batch_size = train_cfg.get("batch_size", 16)
    num_workers = runtime_cfg.get("num_workers", 2)

    train_transform, val_transform = get_transforms(image_size)

    train_dataset = SIDTDDataset(train_samples, transform=train_transform)
    val_dataset = SIDTDDataset(val_samples, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type == "cuda")
    )

    # Model
    model_cfg = cfg.get("model", {})
    model = build_model(
        num_classes=model_cfg.get("num_classes", 2),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = train_cfg.get("learning_rate", 0.0003)
    weight_decay = train_cfg.get("weight_decay", 0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = train_cfg.get("epochs", 25)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler() if (runtime_cfg.get("mixed_precision", True) and device.type == "cuda") else None

    out_cfg = cfg.get("output", {})
    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "models"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / out_cfg.get("checkpoint_filename", "sidtd_efficientnet_b3.pth")

    patience = train_cfg.get("patience", 5)
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info(f"Starting training for {epochs} epochs on {len(train_dataset)} train samples, {len(val_dataset)} val samples...")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_name": "efficientnet_b3",
                "num_classes": 2,
                "class_names": ["BONA_FIDE", "FORGED"],
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": cfg,
            }, checkpoint_path)
            logger.info(f"--> Saved best model checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SIDTD EfficientNet-B3 Model")
    parser.add_argument("--config", type=str, default="training/training_config.yaml", help="Path to training config")
    parser.add_argument("--dataset-root", type=str, default=None, help="Override dataset root path")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu, cuda, mps, auto)")
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        dataset_root_override=args.dataset_root,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        device_override=args.device,
    )
