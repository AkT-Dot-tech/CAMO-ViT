
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformers import SegformerForSemanticSegmentation
from dataset import VIACDataset


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = ROOT / "VIAC" / "train_val_split.csv"
CHECKPOINT_FILE = ROOT / "checkpoints" / "best_model.pth"

OUTPUT_DIR = ROOT / "outputs" / "validation_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

IMAGE_SIZE = 384
NUM_SAMPLES = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


# --------------------------------------------------
# 2. LOAD VALIDATION DATASET
# --------------------------------------------------

print("Loading validation dataset...")

dataset = VIACDataset(
    CSV_FILE,
    split="val",
    image_size=IMAGE_SIZE
)

print(f"Validation samples: {len(dataset)}")


# --------------------------------------------------
# 3. LOAD BEST TRAINED MODEL
# --------------------------------------------------

print("\nLoading best checkpoint...")

if not CHECKPOINT_FILE.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_FILE}"
    )

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
model.to(DEVICE) # type: ignore
model.eval()

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
print(f"Device: {DEVICE}")


# --------------------------------------------------
# 4. PREDICTION VISUALIZATION
# --------------------------------------------------

num_samples = min(NUM_SAMPLES, len(dataset))

# Fixed sample selection for reproducibility
indices = list(range(num_samples))

fig, axes = plt.subplots(
    num_samples,
    4,
    figsize=(16, 4 * num_samples),
    squeeze=False
)

titles = [
    "Original Image",
    "Ground Truth Mask",
    "Predicted Mask",
    "Overlay Comparison"
]

for row, index in enumerate(indices):

    sample = dataset[index]

    image_tensor = sample["pixel_values"]
    mask_tensor = sample["labels"]

    # Add batch dimension and move to GPU
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    # ---------------- INFERENCE ----------------

    with torch.no_grad():

        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(pixel_values=input_tensor)

        logits = torch.nn.functional.interpolate(
            outputs.logits.float(),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        prediction = logits.argmax(dim=1).squeeze(0)

    # ---------------- PREPARE IMAGES ----------------

    # Reverse ImageNet normalization
    image = image_tensor.permute(1, 2, 0).numpy()
    image = (image * STD + MEAN)
    image = np.clip(image, 0, 1)

    ground_truth = mask_tensor.numpy().astype(np.uint8)
    predicted_mask = prediction.cpu().numpy().astype(np.uint8)

    # ---------------- OVERLAY ----------------

    overlay = image.copy()

    # Ground truth: red
    overlay[ground_truth == 1] = (
        0.60 * overlay[ground_truth == 1]
        + 0.40 * np.array([1.0, 0.0, 0.0])
    )

    # Prediction: green
    overlay[predicted_mask == 1] = (
        0.60 * overlay[predicted_mask == 1]
        + 0.40 * np.array([0.0, 1.0, 0.0])
    )

    # Correct overlap: yellow
    overlap = (ground_truth == 1) & (predicted_mask == 1)

    overlay[overlap] = (
        0.60 * image[overlap]
        + 0.40 * np.array([1.0, 1.0, 0.0])
    )

    # ---------------- DISPLAY ----------------

    axes[row, 0].imshow(image)
    axes[row, 1].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
    axes[row, 3].imshow(overlay)

    for col in range(4):
        axes[row, col].axis("off")

        if row == 0:
            axes[row, col].set_title(
                titles[col],
                fontsize=12,
                fontweight="bold"
            )

    # Save individual comparison
    individual_path = OUTPUT_DIR / f"prediction_{index + 1:02d}.png"

    individual_fig, individual_axes = plt.subplots(
        1, 4, figsize=(16, 4)
    )

    individual_images = [
        image,
        ground_truth,
        predicted_mask,
        overlay
    ]

    for ax, visual, title in zip(
        individual_axes,
        individual_images,
        titles
    ):

        if title in ["Ground Truth Mask", "Predicted Mask"]:
            ax.imshow(visual, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(visual)

        ax.set_title(title)
        ax.axis("off")

    individual_fig.tight_layout()
    individual_fig.savefig(
        individual_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close(individual_fig)

    print(f"Saved: {individual_path.name}")


# --------------------------------------------------
# 5. SAVE COMPLETE COMPARISON
# --------------------------------------------------

fig.suptitle(
    "CAMO-ViT — Validation Prediction Results",
    fontsize=18,
    fontweight="bold"
)

fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.99))

comparison_path = OUTPUT_DIR / "all_predictions.png"

fig.savefig(
    comparison_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close(fig)

print("\n" + "=" * 50)
print("PREDICTION VISUALIZATION COMPLETED")
print("=" * 50)

print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Images visualized: {num_samples}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Comparison image: {comparison_path}")
