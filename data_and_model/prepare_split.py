
from pathlib import Path
import csv
import random

ROOT = Path(__file__).resolve().parent.parent / "VIAC"
TRAIN_DIR = ROOT / "traindata"

IMAGE_DIR = TRAIN_DIR / "visible"
MASK_DIR = TRAIN_DIR / "GT"

OUTPUT = ROOT / "train_val_split.csv"

SEED = 42
VAL_RATIO = 0.15

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

images = {
    p.stem: p
    for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
}

masks = {
    p.stem: p
    for p in MASK_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
}

common_ids = sorted(set(images) & set(masks))

if len(common_ids) != 1200:
    raise ValueError(
        f"Expected 1200 matching pairs, found {len(common_ids)}"
    )

random.Random(SEED).shuffle(common_ids)

val_count = int(len(common_ids) * VAL_RATIO)
val_ids = set(common_ids[:val_count])

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "mask", "split"])

    for image_id in sorted(common_ids):
        split = "val" if image_id in val_ids else "train"

        writer.writerow([
            str(images[image_id].as_posix()),
            str(masks[image_id].as_posix()),
            split
        ])

print("Dataset split created successfully!")
print("Training pairs:", len(common_ids) - val_count)
print("Validation pairs:", val_count)
print("Random seed:", SEED)
print("Saved:", OUTPUT)
