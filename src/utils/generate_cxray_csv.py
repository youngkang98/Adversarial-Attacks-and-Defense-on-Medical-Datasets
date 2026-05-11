"""
Generates CXRAY-train.csv and CXRAY-test.csv from the Kaggle Chest X-Ray dataset.

Expected dataset layout (ImageFolder / standard Kaggle structure):
    data/chest_xray/
        train/
            NORMAL/      *.jpeg
            PNEUMONIA/   *.jpeg
        test/
            NORMAL/
            PNEUMONIA/
        val/             (optional, ignored here)

CSV format required by DatasetSeprateByClass:
    Row 0  : file extension   (e.g. jpeg)
    Row 1+ : filename_no_ext , class_name

Usage:
    python src/utils/generate_cxray_csv.py
"""

import sys
import csv
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
import config


def generate_csv(split_dir: Path, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    ext_seen = set()

    for class_folder in sorted(split_dir.iterdir()):
        if not class_folder.is_dir():
            continue
        class_name = class_folder.name
        for img_file in sorted(class_folder.iterdir()):
            if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
                continue
            ext_seen.add(img_file.suffix.lstrip('.').lower())
            rows.append((img_file.stem, class_name))

    if not rows:
        print(f"  WARNING: no images found under {split_dir}")
        return

    # All images in the Kaggle CXRay dataset are .jpeg
    ext = ext_seen.pop() if len(ext_seen) == 1 else 'jpeg'

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ext])           # row 0: extension
        for stem, cls in rows:
            writer.writerow([stem, cls])

    print(f"  Written {len(rows)} entries → {out_csv}")


def main():
    base = config.get_data_path('chest_xray')

    for split in ('train', 'test'):
        split_dir = base / split
        if not split_dir.exists():
            print(f"  SKIP: {split_dir} not found")
            continue
        out_csv = config.get_data_path(f'CXRAY-{split}.csv')
        print(f"Generating {split} CSV …")
        generate_csv(split_dir, out_csv)

    print("Done.")


if __name__ == '__main__':
    main()
