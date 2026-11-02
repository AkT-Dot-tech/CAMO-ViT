
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset import VIACDataset


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TEST_IMAGE_DIR = ROOT / "VIAC" / "testdata" / "visible"
TEST_MASK_DIR = ROOT / "VIAC" / "testdata" / "GT"

CHECKPOINT_FILE = ROOT / "checkpoints" / "best_model.pth"

OUTPUT_DIR = ROOT / "outputs" / "test_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CSV = OUTPUT_DIR / "test_dataset.csv"
PER_IMAGE_RESULTS = OUTPUT_DIR / "per_image_metrics.csv"
SUMMARY_RESULTS = OUTPUT_DIR / "test_metrics_summary.csv"

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

IMAGE_SIZE = 384
BATCH_SIZE = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}


# --------------------------------------------------
# 2. CHECK REQUIRED PATHS
# --------------------------------------------------

print("Preparing untouched test dataset...")

for path in [TEST_IMAGE_DIR, TEST_MASK_DIR, CHECKPOINT_FILE]:

    if not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")


# --------------------------------------------------
# 3. FIND IMAGE AND MASK FILES
# --------------------------------------------------

image_files = sorted(
    p for p in TEST_IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

mask_files = sorted(
    p for p in TEST_MASK_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

if not image_files:
    raise ValueError("No test images found.")

if not mask_files:
    raise ValueError("No test masks found.")

print(f"Found images: {len(image_files)}")
print(f"Found masks:  {len(mask_files)}")


# --------------------------------------------------
# 4. MATCH IMAGES AND MASKS BY FILENAME STEM
# --------------------------------------------------

# Example:
# visible/10.jpg  -> stem = 10
# GT/10.png       -> stem = 10

mask_lookup = {}

for mask_path in mask_files:

    stem = mask_path.stem

    if stem in mask_lookup:
        raise ValueError(
            f"Duplicate mask filename detected: {stem}"
        )

    mask_lookup[stem] = mask_path


image_stems = [p.stem for p in image_files]

if len(image_stems) != len(set(image_stems)):
    raise ValueError(
        "Duplicate image filenames detected after removing extensions."
    )


test_pairs = []

for image_path in image_files:

    if image_path.stem not in mask_lookup:

        raise FileNotFoundError(
            f"Matching mask not found for image: {image_path.name}"
        )

    mask_path = mask_lookup[image_path.stem]

    test_pairs.append(
        (image_path.resolve(), mask_path.resolve())
    )


matched_mask_stems = {mask.stem for _, mask in test_pairs}
unmatched_masks = set(mask_lookup.keys()) - matched_mask_stems

if unmatched_masks:

    raise ValueError(
        f"Unmatched masks detected: {sorted(unmatched_masks)[:10]}"
    )


print(f"Successfully matched pairs: {len(test_pairs)}")


# --------------------------------------------------
# 5. CREATE TEST DATASET CSV
# --------------------------------------------------

with open(TEST_CSV, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=["image", "mask", "split"]
    )

    writer.writeheader()

    for image_path, mask_path in test_pairs:

        writer.writerow({
            "image": str(image_path),
            "mask": str(mask_path),
            "split": "test"
        })


# --------------------------------------------------
# 6. LOAD TEST DATASET
# --------------------------------------------------

test_dataset = VIACDataset(
    TEST_CSV,
    split="test",
    image_size=IMAGE_SIZE
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print(f"\nTest samples: {len(test_dataset)}")
print(f"Device: {DEVICE}")


# --------------------------------------------------
# 7. LOAD BEST TRAINED CHECKPOINT
# --------------------------------------------------

print("\nLoading best trained model...")

checkpoint = torch.load(
    CHECKPOINT_FILE,
    map_location=DEVICE,
    weights_only=True
)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE) # type: ignore
model.eval()

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(
    f"Best validation Dice: "
    f"{checkpoint['best_val_dice']:.4f}"
)


# --------------------------------------------------
# 8. METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(confusion):

    tn, fp = confusion[0]
    fn, tp = confusion[1]

    epsilon = 1e-7

    iou = tp / (
        tp + fp + fn + epsilon
    )

    dice = (2 * tp) / (
        2 * tp + fp + fn + epsilon
    )

    precision = tp / (
        tp + fp + epsilon
    )

    recall = tp / (
        tp + fn + epsilon
    )

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall)
    }


# --------------------------------------------------
# 9. EVALUATE TEST DATASET
# --------------------------------------------------

print("\nStarting final test evaluation...\n")

total_confusion = torch.zeros(
    (2, 2),
    dtype=torch.int64,
    device=DEVICE
)

total_absolute_error = 0.0
total_pixels = 0

per_image_results = []

with torch.no_grad():

    for index, batch in enumerate(test_loader):

        images = batch["pixel_values"].to(DEVICE)
        masks = batch["labels"].to(DEVICE)

        # Model inference
        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(
                pixel_values=images
            )

        # Resize logits to ground-truth resolution
        logits = F.interpolate(
            outputs.logits.float(),
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        # Foreground probabilities
        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        # Binary predictions
        predictions = logits.argmax(dim=1)

        # Pixel-level confusion matrix
        encoded = masks * 2 + predictions

        confusion = torch.bincount(
            encoded.reshape(-1),
            minlength=4
        ).reshape(2, 2)

        total_confusion += confusion

        # MAE using foreground probabilities
        absolute_error = torch.abs(
            probabilities - masks.float()
        )

        image_mae = absolute_error.mean().item()

        total_absolute_error += absolute_error.sum().item()
        total_pixels += masks.numel()

        # Per-image metrics
        image_metrics = calculate_metrics(
            confusion.cpu().tolist()
        )

        image_metrics["image"] = test_pairs[index][0].name
        image_metrics["mae"] = image_mae

        per_image_results.append(image_metrics)

        # Progress
        if (index + 1) % 50 == 0:

            print(
                f"Evaluated {index + 1}/"
                f"{len(test_dataset)} images"
            )


# --------------------------------------------------
# 10. CALCULATE FINAL AGGREGATE METRICS
# --------------------------------------------------

overall_confusion = total_confusion.cpu().tolist()

final_metrics = calculate_metrics(
    overall_confusion
)

final_mae = total_absolute_error / total_pixels

final_metrics["mae"] = final_mae


# --------------------------------------------------
# 11. SAVE PER-IMAGE METRICS
# --------------------------------------------------

with open(
    PER_IMAGE_RESULTS,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "iou",
            "dice",
            "precision",
            "recall",
            "mae"
        ]
    )

    writer.writeheader()
    writer.writerows(per_image_results)


# --------------------------------------------------
# 12. SAVE FINAL SUMMARY
# --------------------------------------------------

with open(
    SUMMARY_RESULTS,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "checkpoint_epoch",
            "test_images",
            "iou",
            "dice",
            "precision",
            "recall",
            "mae"
        ]
    )

    writer.writeheader()

    writer.writerow({
        "checkpoint_epoch": checkpoint["epoch"],
        "test_images": len(test_dataset),
        **final_metrics
    })


# --------------------------------------------------
# 13. PRINT FINAL RESULTS
# --------------------------------------------------

print("\n" + "=" * 55)
print("CAMO-ViT — FINAL TEST EVALUATION")
print("=" * 55)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Test images:      {len(test_dataset)}")

print("\nFINAL TEST METRICS")

print(f"IoU:       {final_metrics['iou']:.4f}")
print(f"Dice:      {final_metrics['dice']:.4f}")
print(f"Precision: {final_metrics['precision']:.4f}")
print(f"Recall:    {final_metrics['recall']:.4f}")
print(f"MAE:       {final_metrics['mae']:.4f}")

print("\nOUTPUT FILES")

print(f"Test dataset:  {TEST_CSV}")
print(f"Summary:       {SUMMARY_RESULTS}")
print(f"Per-image CSV: {PER_IMAGE_RESULTS}")

print("\nFINAL TEST EVALUATION COMPLETED!")
