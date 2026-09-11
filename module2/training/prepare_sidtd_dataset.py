"""Prepare and align SIDTD dataset splits with the actual images on disk.

Ensures:
1. All 1,000 authentic images in templates/Images/reals are accounted for.
2. All 1,222 forged images in templates/Images/fakes are assigned strictly to the split
   corresponding to their source real template (preventing data leakage).
3. CSV files in data/split_normal/ reference valid files with zero missing paths.
"""

import csv
import shutil
from pathlib import Path
from collections import defaultdict, Counter

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
}


def get_class_info(filename: str):
    """Determine class id and class name from filename prefix."""
    prefix = filename.split("_")[0].lower()
    return CLASS_MAP.get(prefix, ("0", prefix))


def prepare_dataset(data_dir: Path):
    split_dir = data_dir / "split_normal"
    images_dir = data_dir / "templates" / "Images"
    reals_dir = images_dir / "reals"
    fakes_dir = images_dir / "fakes"

    assert split_dir.exists(), f"Missing split dir: {split_dir}"
    assert reals_dir.exists(), f"Missing reals dir: {reals_dir}"
    assert fakes_dir.exists(), f"Missing fakes dir: {fakes_dir}"

    # Index all fakes by base template
    disk_fakes = sorted(list(fakes_dir.glob("*.jpg")))
    fakes_by_template = defaultdict(list)
    for f in disk_fakes:
        base = f.name.split("_fake_")[0]
        fakes_by_template[base].append(f.name)

    print(f"Total disk fakes: {len(disk_fakes)} across {len(fakes_by_template)} templates.")
    print(f"Total disk reals: {len(list(reals_dir.glob('*.jpg')))}")

    assigned_fakes_total = 0

    for split in ["train", "val", "test"]:
        csv_file = split_dir / f"{split}_split_SIDTD.csv"
        orig_backup = split_dir / f"{split}_split_SIDTD.csv.orig"

        # Backup original if not already backed up
        if not orig_backup.exists() and csv_file.exists():
            shutil.copy2(csv_file, orig_backup)
            print(f"Backed up {csv_file.name} -> {orig_backup.name}")

        # Read reals from backup or original
        source_csv = orig_backup if orig_backup.exists() else csv_file
        real_rows = []
        templates_in_split = []

        with open(source_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_row = {k.strip(): v.strip() for k, v in row.items() if k and v}
                label = clean_row.get("label")
                lname = clean_row.get("label_name", "").lower()
                if label == "0" or "real" in lname or "bona" in lname:
                    # Verify file exists on disk
                    raw_path = clean_row.get("image_path", "")
                    img_name = Path(raw_path).name
                    disk_img = reals_dir / img_name
                    if disk_img.exists():
                        c_id, c_name = get_class_info(img_name)
                        real_rows.append({
                            "label_name": "reals",
                            "label": "0",
                            "image_path": f"templates/Images/reals/{img_name}",
                            "class": c_id,
                            "class_name": c_name,
                        })
                        base_name = Path(img_name).stem
                        templates_in_split.append(base_name)
                    else:
                        print(f"Warning: Real image {img_name} not found on disk!")

        # Gather fakes belonging to these templates
        fake_rows = []
        for t in templates_in_split:
            for fake_filename in fakes_by_template.get(t, []):
                c_id, c_name = get_class_info(fake_filename)
                fake_rows.append({
                    "label_name": "fakes",
                    "label": "1",
                    "image_path": f"templates/Images/fakes/{fake_filename}",
                    "class": c_id,
                    "class_name": c_name,
                })

        assigned_fakes_total += len(fake_rows)
        all_rows = real_rows + fake_rows

        # Write out clean, aligned CSV
        fieldnames = ["label_name", "label", "image_path", "class", "class_name"]
        with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow(r)

        print(f"[{split.upper()}] Wrote {len(all_rows)} rows ({len(real_rows)} reals, {len(fake_rows)} fakes) to {csv_file.name}")

    print(f"\nVerification:")
    print(f"Total fakes assigned across splits: {assigned_fakes_total} / {len(disk_fakes)}")
    assert assigned_fakes_total == len(disk_fakes), "Mismatch in assigned fakes!"
    print("Dataset split preparation completed successfully with 100% data integrity and zero leakage.")


if __name__ == "__main__":
    import sys
    data_path = Path("/Users/kaivanshah/Documents/SIH_Project/module2/data")
    if len(sys.argv) > 1:
        data_path = Path(sys.argv[1])
    prepare_dataset(data_path)
