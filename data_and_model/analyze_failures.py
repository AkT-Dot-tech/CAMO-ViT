
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import SegformerForSemanticSegmentation
from dataset import VIACDataset


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

TEST_CSV = ROOT / "outputs" / "test_evaluation" / "test_dataset.csv"
METRICS_CSV = ROOT / "outputs" / "test_evaluation" / "per_image_metrics.csv"

CHECKPOINT_FILE = ROOT / "checkpoints" / "best_model.pth"

OUTPUT_DIR = ROOT / "outputs" / "failure_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

IMAGE_SIZE = 384
ZERO_DICE_COUNT = 12
LOW_NONZERO_COUNT = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# --------------------------------------------------
# 2. CHECK FILES
# --------------------------------------------------

print("Starting CAMO-ViT failure analysis...")

for path in [TEST_CSV, METRICS_CSV, CHECKPOINT_FILE]:

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# --------------------------------------------------
# 3. READ PER-IMAGE METRICS
# --------------------------------------------------

with open(METRICS_CSV, "r", newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    metrics = list(reader)

if not metrics:
    raise ValueError("No per-image metrics found.")

for row in metrics:
    row["dice"] = float(row["dice"])
    row["iou"] = float(row["iou"])

# All zero-Dice images
zero_dice = [
    row for row in metrics
    if row["dice"] == 0.0
]

# Lowest-scoring images with non-zero Dice
nonzero_dice = [
    row for row in metrics
    if row["dice"] > 0.0   # type: ignore
]

nonzero_dice.sort(key=lambda row: row["dice"])

selected = (
    zero_dice[:ZERO_DICE_COUNT]
    + nonzero_dice[:LOW_NONZERO_COUNT]
)

# Avoid duplicate filenames
selected_by_name = {
    row["image"]: row for row in selected
}

selected = list(selected_by_name.values())

print(f"Total test images: {len(metrics)}")
print(f"Zero-Dice images found: {len(zero_dice)}")
print(f"Selected images for analysis: {len(selected)}")


# --------------------------------------------------
# 4. LOAD TEST DATASET
# --------------------------------------------------

print("\nLoading test dataset...")

dataset = VIACDataset(
    TEST_CSV,
    split="test",
    image_size=IMAGE_SIZE
)

# Build filename-to-index mapping using dataset paths
dataset_lookup = {}

for index, sample_info in enumerate(dataset.samples):
    image_name = sample_info["image"].name
    dataset_lookup[image_name] = index

for row in selected:
    if row["image"] not in dataset_lookup:
        raise FileNotFoundError(
            f"Image not found in test dataset: {row['image']}"
        )


# --------------------------------------------------
# 5. LOAD BEST MODEL
# --------------------------------------------------

print("\nLoading best checkpoint...")

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

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)   # type: ignore
model.eval()

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
print(f"Device: {DEVICE}")


# --------------------------------------------------
# 6. HELPER: PREPARE OVERLAY
# --------------------------------------------------

def create_overlay(image, ground_truth, prediction):

    overlay = image.copy()

    # Red: ground truth only (false negatives)
    false_negative = (
        (ground_truth == 1) & (prediction == 0)
    )

    # Green: prediction only (false positives)
    false_positive = (
        (ground_truth == 0) & (prediction == 1)
    )

    # Yellow: correct foreground overlap
    correct_overlap = (
        (ground_truth == 1) & (prediction == 1)
    )

    overlay[false_negative] = (
        0.55 * image[false_negative]
        + 0.45 * np.array([1.0, 0.0, 0.0])
    )

    overlay[false_positive] = (
        0.55 * image[false_positive]
        + 0.45 * np.array([0.0, 1.0, 0.0])
    )

    overlay[correct_overlap] = (
        0.55 * image[correct_overlap]
        + 0.45 * np.array([1.0, 1.0, 0.0])
    )

    return np.clip(overlay, 0, 1)


# --------------------------------------------------
# 7. GENERATE PREDICTIONS AND SAVE COMPARISONS
# --------------------------------------------------

print("\nGenerating failure-case visualizations...\n")

individual_dir = OUTPUT_DIR / "individual"
individual_dir.mkdir(exist_ok=True)

results = []

with torch.no_grad():

    for count, row in enumerate(selected, start=1):

        image_name = row["image"]
        dataset_index = dataset_lookup[image_name]

        sample = dataset[dataset_index]

        image_tensor = sample["pixel_values"]
        mask_tensor = sample["labels"]

        input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        # Model inference
        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(pixel_values=input_tensor)

        logits = F.interpolate(
            outputs.logits.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        prediction = logits.argmax(dim=1).squeeze(0)

        # Reverse ImageNet normalization
        image = image_tensor.permute(1, 2, 0).numpy()
        image = np.clip(image * STD + MEAN, 0, 1)

        ground_truth = mask_tensor.numpy().astype(np.uint8)
        predicted_mask = prediction.cpu().numpy().astype(np.uint8)

        overlay = create_overlay(
            image,
            ground_truth,
            predicted_mask
        )

        # Determine category
        category = (
            "zero_dice"
            if row["dice"] == 0.0
            else "low_dice"
        )

        # Save individual comparison
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        visuals = [
            image,
            ground_truth,
            predicted_mask,
            overlay
        ]

        titles = [
            "Original Image",
            "Ground Truth",
            "Prediction",
            "Overlay"
        ]

        for ax, visual, title in zip(axes, visuals, titles):

            if title in ["Ground Truth", "Prediction"]:
                ax.imshow(visual, cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(visual)

            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(
            f"{image_name} | Dice: {row['dice']:.4f} | "
            f"IoU: {row['iou']:.4f}",
            fontsize=13
        )

        fig.tight_layout()

        output_path = (
            individual_dir
            / f"{category}_{count:02d}_{Path(image_name).stem}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        results.append({
            "image": image_name,
            "dice": row["dice"],
            "iou": row["iou"],
            "category": category,
            "visualization": str(output_path)
        })

        print(
            f"[{count}/{len(selected)}] "
            f"{image_name} | Dice: {row['dice']:.4f}"
        )


# --------------------------------------------------
# 8. CREATE OVERVIEW GRID
# --------------------------------------------------

print("\nCreating overview grid...")

columns = 2
rows_count = (len(results) + columns - 1) // columns

fig, axes = plt.subplots(
    rows_count,
    columns,
    figsize=(18, rows_count * 4.5),
    squeeze=False
)

for ax, result in zip(axes.flat, results):

    image_path = Path(result["visualization"])
    visual = plt.imread(image_path)

    ax.imshow(visual)
    ax.set_title(
        f"{result['image']} | Dice: {result['dice']:.4f}",
        fontsize=10
    )
    ax.axis("off")

for ax in axes.flat[len(results):]:
    ax.axis("off")

fig.suptitle(
    "CAMO-ViT — Lowest Performing Test Images",
    fontsize=17,
    fontweight="bold"
)

fig.tight_layout(rect=(0, 0, 1, 0.99))

overview_path = OUTPUT_DIR / "failure_analysis_overview.png"

fig.savefig(
    overview_path,
    dpi=120,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# 9. SAVE SELECTED IMAGE LIST
# --------------------------------------------------

selection_csv = OUTPUT_DIR / "selected_failure_cases.csv"

with open(
    selection_csv,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "image",
            "dice",
            "iou",
            "category",
            "visualization"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# --------------------------------------------------
# 10. FINISH
# --------------------------------------------------

print("\n" + "=" * 55)
print("FAILURE ANALYSIS COMPLETED")
print("=" * 55)

print(f"Images analysed: {len(results)}")
print(f"Zero-Dice cases: {sum(r['category'] == 'zero_dice' for r in results)}")
print(f"Low-Dice cases:  {sum(r['category'] == 'low_dice' for r in results)}")

print(f"\nOverview: {overview_path}")
print(f"Individual images: {individual_dir}")
print(f"Selected cases CSV: {selection_csv}")
