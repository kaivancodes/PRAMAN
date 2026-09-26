"""Prepare and align SIDTD dataset splits with the actual images on disk.

Ensures:
1. All authentic images in templates/Images/reals are accounted for (including newly added images).
2. All forged images in templates/Images/fakes are assigned strictly to the split
   corresponding to their source real template (preventing data leakage).
3. Any new custom images (e.g. Indian passports, IDs) added to reals/ or fakes/
   are automatically detected, assigned to train/val splits (80/20 ratio), and recorded in CSVs.
4. Provides a CLI to easily add new real/fake document images into the dataset.
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

CLASS_MAP = {
    "alb": ("0", "alb"),
    "aze": ("1", "aze"),
    "esp": ("2", "esp"),
    "est": ("3", "est"),
    "fin": ("4", "fin"),
    "grc": ("5", "grc"),
    "lva": ("6", "lva"),
    "rus": ("7", "rus"),
    "srb": ("8", "srb"),
    "svk": ("9", "svk"),
    "ind": ("10", "ind"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def get_class_info(filename: str) -> Tuple[str, str]:
    """Determine class id and class name from filename prefix."""
    prefix = filename.split("_")[0].lower()
    if prefix not in CLASS_MAP:
        new_id = str(len(CLASS_MAP))
        CLASS_MAP[prefix] = (new_id, prefix)
    return CLASS_MAP[prefix]


def find_image_files(directory: Path) -> List[Path]:
    """Find all valid image files with supported extensions."""
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(list(set(files)))


def prepare_dataset(data_dir: Path):
    """Scan disk, preserve existing splits, and assign any new images into train/val."""
    split_dir = data_dir / "split_normal"
    images_dir = data_dir / "templates" / "Images"
    reals_dir = images_dir / "reals"
    fakes_dir = images_dir / "fakes"

    assert split_dir.exists(), f"Missing split dir: {split_dir}"
    assert reals_dir.exists(), f"Missing reals dir: {reals_dir}"
    assert fakes_dir.exists(), f"Missing fakes dir: {fakes_dir}"

    disk_reals = find_image_files(reals_dir)
    disk_fakes = find_image_files(fakes_dir)

    print(f"[*] Total authentic images on disk: {len(disk_reals)}")
    print(f"[*] Total forged images on disk   : {len(disk_fakes)}")

    # Index all fakes by base template
    fakes_by_template = defaultdict(list)
    unmatched_fakes = []
    disk_real_stems = {r.stem: r for r in disk_reals}

    for f in disk_fakes:
        if "_fake_" in f.name:
            base = f.name.split("_fake_")[0]
        else:
            base = f.stem
        fakes_by_template[base].append(f.name)

    # 1. Read existing assignments from CSVs
    split_assigned_reals: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}
    split_templates: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    already_indexed_reals: Set[str] = set()

    for split in ["train", "val", "test"]:
        csv_file = split_dir / f"{split}_split_SIDTD.csv"
        if not csv_file.exists():
            continue

        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k and v}
                label = clean_row.get("label")
                lname = clean_row.get("label_name", "").lower()
                if label == "0" or "real" in lname or "bona" in lname:
                    raw_path = clean_row.get("image_path", "")
                    img_name = Path(raw_path).name
                    disk_img = reals_dir / img_name
                    if disk_img.exists() and img_name not in already_indexed_reals:
                        c_id, c_name = get_class_info(img_name)
                        split_assigned_reals[split].append({
                            "label_name": "reals",
                            "label": "0",
                            "image_path": f"templates/Images/reals/{img_name}",
                            "class": c_id,
                            "class_name": c_name,
                        })
                        split_templates[split].append(disk_img.stem)
                        already_indexed_reals.add(img_name)

    # 2. Check for NEW, unassigned real images on disk
    new_reals = [r for r in disk_reals if r.name not in already_indexed_reals]
    if new_reals:
        print(f"\n[+] Detected {len(new_reals)} NEW unassigned authentic document(s):")
        for nr in new_reals:
            print(f"    - {nr.name}")

        # Distribute new reals (80% train, 20% val)
        random.seed(42)
        random.shuffle(new_reals)
        for i, r in enumerate(new_reals):
            # Put 80% in train (items 0, 1, 2, 3) and 20% in val (item 4, 9, etc.)
            target_split = "val" if (i % 5 == 4) else "train"
            c_id, c_name = get_class_info(r.name)
            split_assigned_reals[target_split].append({
                "label_name": "reals",
                "label": "0",
                "image_path": f"templates/Images/reals/{r.name}",
                "class": c_id,
                "class_name": c_name,
            })
            split_templates[target_split].append(r.stem)
            already_indexed_reals.add(r.name)

    # 3. Assign fakes to corresponding splits (strictly zero data-leakage)
    assigned_fake_names: Set[str] = set()
    split_assigned_fakes: Dict[str, List[Dict[str, str]]] = {"train": [], "val": [], "test": []}

    # First, assign paired fakes that belong to known templates
    for split in ["train", "val", "test"]:
        for t in split_templates[split]:
            for fake_filename in fakes_by_template.get(t, []):
                if fake_filename not in assigned_fake_names:
                    c_id, c_name = get_class_info(fake_filename)
                    split_assigned_fakes[split].append({
                        "label_name": "fakes",
                        "label": "1",
                        "image_path": f"templates/Images/fakes/{fake_filename}",
                        "class": c_id,
                        "class_name": c_name,
                    })
                    assigned_fake_names.add(fake_filename)

    # Second, assign standalone / unpaired fakes (80% train, 20% val)
    standalone_fakes = [f for f in disk_fakes if f.name not in assigned_fake_names]
    if standalone_fakes:
        print(f"\n[+] Detected {len(standalone_fakes)} standalone forged document(s) -> Partitioning 80/20 into train/val:")
        for sf in standalone_fakes[:5]:
            print(f"    - {sf.name}")
        if len(standalone_fakes) > 5:
            print(f"    ... and {len(standalone_fakes) - 5} more")

        random.seed(42)
        random.shuffle(standalone_fakes)
        for i, sf in enumerate(standalone_fakes):
            target_split = "val" if (i % 5 == 4) else "train"
            c_id, c_name = get_class_info(sf.name)
            split_assigned_fakes[target_split].append({
                "label_name": "fakes",
                "label": "1",
                "image_path": f"templates/Images/fakes/{sf.name}",
                "class": c_id,
                "class_name": c_name,
            })
            assigned_fake_names.add(sf.name)

    # 4. Write out updated CSV splits
    total_reals_indexed = 0
    total_fakes_indexed = 0
    for split in ["train", "val", "test"]:
        csv_file = split_dir / f"{split}_split_SIDTD.csv"
        real_rows = split_assigned_reals[split]
        fake_rows = split_assigned_fakes[split]
        all_rows = real_rows + fake_rows

        total_reals_indexed += len(real_rows)
        total_fakes_indexed += len(fake_rows)

        fieldnames = ["label_name", "label", "image_path", "class", "class_name"]
        with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)

        print(f"[{split.upper()}] Total: {len(all_rows):<5} (Reals: {len(real_rows):<4}, Fakes: {len(fake_rows):<4}) -> {csv_file.name}")

    print(f"\n[+] Dataset split preparation complete.")
    print(f"    Total authentic documents indexed: {total_reals_indexed} / {len(disk_reals)}")
    print(f"    Total forged documents indexed   : {total_fakes_indexed} / {len(disk_fakes)}")


def add_document_image(
    source_image: Path,
    label: str,
    country: str,
    doc_type: str,
    data_dir: Path,
    custom_name: str = None,
) -> Path:
    """Helper to copy a raw image into templates/Images/reals or fakes and re-align splits."""
    if not source_image.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")

    country = country.lower().strip()
    doc_type = doc_type.lower().strip()
    label = label.lower().strip()

    images_dir = data_dir / "templates" / "Images"
    dest_sub = "reals" if label in ("0", "real", "bona_fide") else "fakes"
    dest_dir = images_dir / dest_sub
    dest_dir.mkdir(parents=True, exist_ok=True)

    ext = source_image.suffix.lower()
    if ext not in IMAGE_EXTENSIONS:
        ext = ".jpg"

    if custom_name:
        dest_filename = f"{custom_name}{ext}"
    else:
        existing = list(dest_dir.glob(f"{country}_{doc_type}_*{ext}"))
        idx = len(existing)
        dest_filename = f"{country}_{doc_type}_{idx:02d}{ext}"

    dest_path = dest_dir / dest_filename
    shutil.copy2(source_image, dest_path)
    print(f"[+] Added {source_image.name} -> {dest_path.relative_to(data_dir)}")

    # Update dataset splits
    prepare_dataset(data_dir)
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="SIDTD Dataset Split Preparation & Ingestion")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent.parent / "data"), help="Path to data directory")
    parser.add_argument("--add-real", type=str, default=None, help="Path to authentic document image to add")
    parser.add_argument("--add-fake", type=str, default=None, help="Path to forged document image to add")
    parser.add_argument("--country", type=str, default="ind", help="Country prefix (e.g. ind, esp, aze)")
    parser.add_argument("--doc-type", type=str, default="passport", choices=["passport", "id", "visa", "driving_license", "permit"])
    args = parser.parse_args()

    data_path = Path(args.data_dir).resolve()

    if args.add_real:
        add_document_image(
            source_image=Path(args.add_real),
            label="real",
            country=args.country,
            doc_type=args.doc_type,
            data_dir=data_path,
        )
    elif args.add_fake:
        add_document_image(
            source_image=Path(args.add_fake),
            label="fake",
            country=args.country,
            doc_type=args.doc_type,
            data_dir=data_path,
        )
    else:
        prepare_dataset(data_path)


if __name__ == "__main__":
    main()
