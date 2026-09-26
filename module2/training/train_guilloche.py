"""GPU training for PRAMAN Guilloché / Security-Pattern Forensics.

Training input:
    dataset image -> pattern extraction -> 256x256 pattern crop
    -> resize 128x128 -> ImageNet normalization -> Siamese ResNet-18

No production upload-quality validation is used here. The training image
enters the Guilloché-specific preprocessing directly.
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from module2.models.siamese_guilloche import SiameseResNet18
from module2.guilloche.pattern_extractor import PatternExtractor
from module2.preprocessing.raw_preprocessor import RawDocumentPreprocessor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [Guilloche-Train] %(message)s")
logger = logging.getLogger("guilloche_train")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_images(root: Path) -> List[Path]:
    return [
        p for p in root.glob("**/*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    ]


def template_key(path: Path) -> str:
    # SIDTD forged files normally preserve the source/template relation in metadata;
    # filename stem is the safe fallback for grouping without cross-template leakage.
    stem = path.stem.lower()
    if "_fake_" in stem:
        return stem.split("_fake_")[0]
    return stem


def build_pairs(images: List[Path], pairs_per_epoch: int, seed: int) -> List[Tuple[Path, Path, int]]:
    rng = random.Random(seed)
    groups = {}
    for p in images:
        groups.setdefault(template_key(p), []).append(p)

    keys = list(groups)
    pairs = []
    if not keys:
        return pairs

    # Positive: two views/images from the same template/group.
    for _ in range(pairs_per_epoch // 2):
        k = rng.choice(keys)
        items = groups[k]
        p1 = rng.choice(items)
        p2 = rng.choice(items)
        pairs.append((p1, p2, 1))

    # Negative: different template groups.
    for _ in range(pairs_per_epoch - len(pairs)):
        k1, k2 = rng.sample(keys, 2) if len(keys) > 1 else (keys[0], keys[0])
        while k2 == k1 and len(keys) > 1:
            k2 = rng.choice(keys)
        pairs.append((rng.choice(groups[k1]), rng.choice(groups[k2]), -1))

    rng.shuffle(pairs)
    return pairs


class GuillochePairDataset(Dataset):
    def __init__(self, pairs, image_size=128, train=False):
        self.pairs = pairs
        self.extractor = PatternExtractor(crop_size=(256, 256))
        self.image_size = image_size
        self.train = train
        self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.raw_preprocessor = RawDocumentPreprocessor()

    def _load(self, path):
        image = Image.open(path).convert("RGB")
        # Use the same common document normalization as production before the
        # Guilloché-specific branch begins.
        image, _ = self.raw_preprocessor.preprocess_raw_document(image, doc_type="training_document")
        # Guilloché-specific preprocessing begins with pattern extraction.
        crop = self.extractor.extract_pattern(image)
        crop = crop.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        if self.train:
            if random.random() < 0.5:
                crop = transforms.functional.hflip(crop)
            if random.random() < 0.3:
                crop = transforms.functional.adjust_contrast(crop, 1.10)
        tensor = transforms.ToTensor()(crop)
        return self.norm(tensor)

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        return self._load(p1), self._load(p2), torch.tensor(float(label))


def split_groups(images, seed=42):
    groups = {}
    for p in images:
        groups.setdefault(template_key(p), []).append(p)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    cut = max(1, int(len(keys) * 0.8))
    train_keys, val_keys = keys[:cut], keys[cut:]
    return [p for k in train_keys for p in groups[k]], [p for k in val_keys for p in groups[k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="training/guilloche_training_config.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg["runtime"]["seed"])
    device_name = args.device or cfg["runtime"]["device"]
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name if device_name != "auto" else "cpu")
    logger.info("Training device: %s", device)

    root = Path(cfg["data"]["dataset_root"])
    images = []
    for sub in [
        root / "templates" / "Images" / "reals",
        root / "templates" / "Images" / "fakes",
        root / "MIDV-2020",
    ]:
        if sub.exists():
            images.extend(collect_images(sub))

    train_images, val_images = split_groups(images, cfg["runtime"]["seed"])
    pairs_n = 256 if args.dry_run else cfg["data"]["pairs_per_epoch"]
    train_pairs = build_pairs(train_images, pairs_n, cfg["runtime"]["seed"])
    val_pairs = build_pairs(val_images, max(256, pairs_n // 4), cfg["runtime"]["seed"] + 1)

    train_ds = GuillochePairDataset(train_pairs, 128, True)
    val_ds = GuillochePairDataset(val_pairs, 128, False)
    bs = args.batch_size or cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=cfg["runtime"]["num_workers"], pin_memory=device.type=="cuda")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=cfg["runtime"]["num_workers"], pin_memory=device.type=="cuda")

    model = SiameseResNet18(
        embedding_dim=cfg["model"]["embedding_dim"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)
    criterion = nn.CosineEmbeddingLoss(margin=cfg["training"]["margin"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["runtime"]["mixed_precision"] and device.type=="cuda")

    epochs = 1 if args.dry_run else (args.epochs or cfg["training"]["epochs"])
    best_loss = float("inf")
    history = []
    out_dir = Path(cfg["output"]["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                ea, eb = model(a, b)
                loss = criterion(ea, eb, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for a, b, y in val_loader:
                a, b, y = a.to(device), b.to(device), y.to(device)
                ea, eb = model(a, b)
                val_losses.append(float(criterion(ea, eb, y).item()))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss}
        history.append(row)
        logger.info("Epoch %d/%d | train_loss=%.4f | val_loss=%.4f", epoch, epochs, row["train_loss"], val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "model_name": "resnet18",
                "embedding_dim": cfg["model"]["embedding_dim"],
                "input": "Guilloche pattern crop",
                "pattern_crop_size": 256,
                "model_input_size": 128,
                "normalization": "ImageNet",
                "similarity_threshold": cfg["output"]["similarity_threshold"],
                "val_metrics": row,
                "training_config": cfg,
            }
            torch.save(checkpoint, out_dir / cfg["output"]["checkpoint_filename"])
            (out_dir / "guilloche_training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    logger.info("Best checkpoint: %s", out_dir / cfg["output"]["checkpoint_filename"])


if __name__ == "__main__":
    main()
