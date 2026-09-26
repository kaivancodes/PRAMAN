"""Training script for Guilloché and Security Pattern Verification using Siamese ResNet-18.

Trains a Siamese metric-learning model with Cosine Embedding Loss on security pattern pairs:
- Positive pairs (Target = +1): Matching authentic security background patterns from the same document template.
- Negative pairs (Target = -1): Authentic background patterns paired with forged patterns or different document types.

HPC-ready: Supports CUDA, MPS, mixed precision, LR scheduling, and experiment logging.
"""

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import yaml
from sklearn.metrics import roc_auc_score

# Ensure src/ is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from module2.models.siamese_guilloche import SiameseResNet18
from module2.preprocessing.raw_preprocessor import RawDocumentPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Guilloche-Train]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("guilloche_train")


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GuillochePairDataset(Dataset):
    """Pairwise dataset for Siamese security pattern verification.

    Returns:
        img1: Tensor (3, H, W)
        img2: Tensor (3, H, W)
        target: float (+1.0 for matching/authentic, -1.0 for non-matching/forged)
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str, int]],
        transform1: Optional[transforms.Compose] = None,
        transform2: Optional[transforms.Compose] = None,
        enable_raw_preprocessing: bool = True,
    ):
        """Initialize dataset.

        Args:
            pairs: List of (path1, path2, label) where label is +1 (match) or -1 (mismatch).
            transform1: Transformations for first image.
            transform2: Transformations for second image.
            enable_raw_preprocessing: Whether to apply boundary rectification & CLAHE normalization.
        """
        self.pairs = pairs
        self.transform1 = transform1
        self.transform2 = transform2 or transform1
        self.enable_raw_preprocessing = enable_raw_preprocessing
        self.preprocessor = RawDocumentPreprocessor() if enable_raw_preprocessing else None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1, p2, label = self.pairs[idx]
        try:
            with open(p1, "rb") as f:
                img1 = Image.open(f).convert("RGB")
                if self.enable_raw_preprocessing and self.preprocessor is not None:
                    try:
                        clean1, _ = self.preprocessor.preprocess_raw_document(img1)
                        img1 = clean1
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to open image {p1}: {e}")
            img1 = Image.new("RGB", (128, 128), color=0)

        try:
            with open(p2, "rb") as f:
                img2 = Image.open(f).convert("RGB")
                if self.enable_raw_preprocessing and self.preprocessor is not None:
                    try:
                        clean2, _ = self.preprocessor.preprocess_raw_document(img2)
                        img2 = clean2
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to open image {p2}: {e}")
            img2 = Image.new("RGB", (128, 128), color=0)

        if self.transform1:
            img1 = self.transform1(img1)
        if self.transform2:
            img2 = self.transform2(img2)

        return img1, img2, torch.tensor(float(label), dtype=torch.float32)


def get_transforms(image_size: int = 128) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms for pattern crops."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 16, image_size + 16)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=6),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor],
    )
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transform, val_transform


def _images(root: Path) -> List[str]:
    out: List[str] = []
    if not root.exists():
        return out
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        out.extend(str(p) for p in root.rglob(ext))
    return sorted(set(out))


def _load_sidtd_template_images(dataset_root: Path) -> Tuple[List[str], List[str]]:
    real_dir = dataset_root / "templates" / "Images" / "reals"
    fake_dir = dataset_root / "templates" / "Images" / "fakes"
    return _images(real_dir), _images(fake_dir)


def _load_midv_images(midv_root: Path) -> List[str]:
    """Load MIDV-2020 document/template imagery as authentic reference images."""
    candidates = [
        midv_root / "templates" / "images",
        midv_root / "dataset" / "templates" / "images",
        midv_root / "dataset" / "images",
    ]
    found: List[str] = []
    for root in candidates:
        found.extend(_images(root))
    return sorted(set(found))


def _group_key(path: str) -> str:
    """Group related captures/templates without treating filenames as labels."""
    stem = Path(path).stem.lower()
    parts = stem.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0]


def build_pairs_from_disk(
    dataset_root: Path,
    pairs_count: int = 2000,
    midv_root: Optional[Path] = None,
) -> List[Tuple[str, str, int]]:
    """Build Guilloché metric-learning pairs from MIDV-2020 + SIDTD.

    MIDV-2020 contributes bona-fide document/template images. SIDTD contributes
    bona-fide and forged template images. No filename is interpreted as a
    tamper label. Positive pairs are from the same template/group; negative
    pairs are from different groups or bona-fide/forged SIDTD patterns.
    """
    sidtd_reals, sidtd_fakes = _load_sidtd_template_images(dataset_root)
    midv_reals = _load_midv_images(midv_root) if midv_root else []

    real_images = sorted(set(sidtd_reals + midv_reals))
    fake_images = sidtd_fakes

    if len(real_images) < 2:
        logger.warning("Need at least two authentic MIDV/SIDTD images for Guilloché training.")
        return []

    groups: Dict[str, List[str]] = {}
    for p in real_images:
        groups.setdefault(_group_key(p), []).append(p)

    group_keys = list(groups)
    positive_candidates = [g for g in group_keys if len(groups[g]) >= 2]

    pairs: List[Tuple[str, str, int]] = []
    half = pairs_count // 2

    # Positive: same document/template family.
    for _ in range(half):
        if positive_candidates:
            g = random.choice(positive_candidates)
            a, b = random.sample(groups[g], 2)
        else:
            a = b = random.choice(real_images)
        pairs.append((a, b, 1))

    # Negative: forged-vs-real where available, otherwise different authentic groups.
    for _ in range(pairs_count - half):
        if fake_images and random.random() < 0.6:
            pairs.append((random.choice(real_images), random.choice(fake_images), -1))
        elif len(group_keys) >= 2:
            g1, g2 = random.sample(group_keys, 2)
            pairs.append((random.choice(groups[g1]), random.choice(groups[g2]), -1))
        else:
            pairs.append((random.choice(real_images), random.choice(real_images), -1))

    random.shuffle(pairs)
    logger.info(
        "Guilloché sources: MIDV authentic=%d, SIDTD real=%d, SIDTD forged=%d; pairs=%d",
        len(midv_reals), len(sidtd_reals), len(sidtd_fakes), len(pairs)
    )
    return pairs


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    use_amp: bool,
) -> float:
    """Train Siamese model for one epoch."""
    model.train()
    total_loss = 0.0
    steps = 0

    for img1, img2, target in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None and device.type == "cuda":
            with torch.cuda.amp.autocast():
                emb1 = model.extract_features(img1)
                emb2 = model.extract_features(img2)
                loss = criterion(emb1, emb2, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            emb1 = model.extract_features(img1)
            emb2 = model.extract_features(img2)
            loss = criterion(emb1, emb2, target)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(1, steps)


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate Siamese model on validation pairs."""
    model.eval()
    total_loss = 0.0
    steps = 0

    all_sims: List[float] = []
    all_targets: List[int] = []

    pos_sims: List[float] = []
    neg_sims: List[float] = []

    for img1, img2, target in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        target = target.to(device)

        emb1 = model.extract_features(img1)
        emb2 = model.extract_features(img2)
        loss = criterion(emb1, emb2, target)

        total_loss += loss.item()
        steps += 1

        sims = F.cosine_similarity(emb1, emb2).cpu().numpy()
        targets_np = target.cpu().numpy()

        for s, t in zip(sims, targets_np):
            all_sims.append(float(s))
            all_targets.append(1 if t > 0 else 0)
            if t > 0:
                pos_sims.append(float(s))
            else:
                neg_sims.append(float(s))

    val_loss = total_loss / max(1, steps)
    mean_pos = float(np.mean(pos_sims)) if pos_sims else 0.0
    mean_neg = float(np.mean(neg_sims)) if neg_sims else 0.0
    separation = mean_pos - mean_neg

    # Accuracy at 0.5 cosine similarity
    correct = sum((s >= 0.5) == (t == 1) for s, t in zip(all_sims, all_targets))
    accuracy = correct / max(1, len(all_targets))

    # ROC-AUC if both classes present
    auc = 0.5
    if len(set(all_targets)) > 1:
        try:
            auc = float(roc_auc_score(all_targets, all_sims))
        except Exception:
            auc = 0.5

    return {
        "val_loss": val_loss,
        "val_accuracy": accuracy,
        "val_auc": auc,
        "mean_positive_sim": mean_pos,
        "mean_negative_sim": mean_neg,
        "separation_margin": separation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Guilloché Siamese ResNet-18 Model")
    parser.add_argument("--config", type=str, default="training/guilloche_training_config.yaml", help="Path to config YAML")
    parser.add_argument("--dataset-root", type=str, default=None, help="Root path of dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint file")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable raw document preprocessing during training")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 quick epoch to verify pipeline")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        # Fallback to local
        config_path = Path(__file__).resolve().parent / "guilloche_training_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    set_seed(cfg.get("runtime", {}).get("seed", 42))

    # Resolve device
    dev_str = cfg.get("runtime", {}).get("device", "auto")
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

    # Dataset params
    dataset_root_str = args.dataset_root or cfg.get("data", {}).get("dataset_root", "data")
    dataset_root = Path(dataset_root_str)
    if not dataset_root.is_absolute():
        dataset_root = Path(__file__).resolve().parent.parent / dataset_root_str

    pairs_count = 50 if args.dry_run else cfg.get("data", {}).get("pairs_per_epoch", 1000)
    midv_root_str = cfg.get("data", {}).get("midv_root", "data/MIDV-2020")
    midv_root = Path(midv_root_str)
    if not midv_root.is_absolute():
        midv_root = Path(__file__).resolve().parent.parent / midv_root_str
    all_pairs = build_pairs_from_disk(dataset_root, pairs_count=pairs_count, midv_root=midv_root)

    if not all_pairs:
        logger.error(f"No pairs could be created from {dataset_root}. Exiting.")
        sys.exit(1)

    # Train / Val Split (80/20)
    val_ratio = cfg.get("data", {}).get("val_split", 0.2)
    split_idx = int(len(all_pairs) * (1.0 - val_ratio))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]

    img_size = cfg.get("training", {}).get("image_size", 128)
    train_tf, val_tf = get_transforms(image_size=img_size)

    enable_raw_prep = not args.no_preprocess and cfg.get("preprocessing", {}).get("enabled", True)
    logger.info(f"Raw Document Preprocessing for Guilloché training: {'ENABLED' if enable_raw_prep else 'DISABLED'}")

    train_ds = GuillochePairDataset(train_pairs, transform1=train_tf, enable_raw_preprocessing=enable_raw_prep)
    val_ds = GuillochePairDataset(val_pairs, transform1=val_tf, enable_raw_preprocessing=enable_raw_prep)

    batch_size = args.batch_size or (4 if args.dry_run else cfg.get("training", {}).get("batch_size", 32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build Model
    emb_dim = cfg.get("model", {}).get("embedding_dim", 128)
    pretrained = cfg.get("model", {}).get("pretrained", True)
    model = SiameseResNet18(embedding_dim=emb_dim, pretrained=pretrained)
    model.to(device)

    # Optimizer & Loss
    lr = args.lr or cfg.get("training", {}).get("learning_rate", 0.0002)
    weight_decay = cfg.get("training", {}).get("weight_decay", 0.0001)
    margin = cfg.get("training", {}).get("margin", 0.3)

    criterion = nn.CosineEmbeddingLoss(margin=margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 1 if args.dry_run else (args.epochs or cfg.get("training", {}).get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp = cfg.get("runtime", {}).get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Output setup
    out_dir = Path(__file__).resolve().parent.parent / cfg.get("output", {}).get("checkpoint_dir", "models")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = args.output or cfg.get("output", {}).get("checkpoint_filename", "guilloche_siamese_resnet18.pth")
    out_path = out_dir / Path(out_filename).name

    best_auc = 0.0
    best_loss = float("inf")
    patience = cfg.get("training", {}).get("patience", 6)
    patience_counter = 0

    logger.info(f"Starting Siamese ResNet-18 Guilloché training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        scheduler.step()

        val_metrics = validate(model, val_loader, criterion, device)
        v_loss = val_metrics["val_loss"]
        v_auc = val_metrics["val_auc"]
        v_acc = val_metrics["val_accuracy"]
        pos_sim = val_metrics["mean_positive_sim"]
        neg_sim = val_metrics["mean_negative_sim"]

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} | "
            f"Val Acc: {v_acc:.4f} | "
            f"Val AUC: {v_auc:.4f} | "
            f"Pos Sim: {pos_sim:.4f} | "
            f"Neg Sim: {neg_sim:.4f}"
        )

        is_best = v_auc > best_auc or (v_auc == best_auc and v_loss < best_loss)
        if is_best:
            best_auc = v_auc
            best_loss = v_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_name": "resnet18",
                "embedding_dim": emb_dim,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "training_config": cfg,
            }
            if cfg.get("output", {}).get("save_optimizer", False):
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(checkpoint, out_path)
            logger.info(f"[*] Best model checkpoint saved to: {out_path} (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    logger.info(f"Guilloché training complete. Best checkpoint saved at: {out_path}")


if __name__ == "__main__":
    main()
