
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Dataset paths
ROOT = Path(__file__).resolve().parent.parent / "VIAC"
IMAGE_DIR = ROOT / "traindata" / "visible"
MASK_DIR = ROOT / "traindata" / "GT"

# Number of samples
NUM_SAMPLES = 6
random.seed(42)

# Find matching image-mask pairs
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

images = {
    p.stem: p for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
}

masks = {
    p.stem: p for p in MASK_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
}

common_ids = sorted(set(images) & set(masks))

if not common_ids:
    raise RuntimeError("No matching image-mask pairs found.")

selected = random.sample(
    common_ids,
    min(NUM_SAMPLES, len(common_ids))
)

# Display samples
fig, axes = plt.subplots(
    len(selected), 3,
    figsize=(14, 4 * len(selected))
)

if len(selected) == 1:
    axes = np.expand_dims(axes, axis=0)

for row, image_id in enumerate(selected):

    # Load RGB image
    image = Image.open(images[image_id]).convert("RGB")
    image = np.array(image)

    # Load ground-truth mask
    mask = Image.open(masks[image_id]).convert("L")
    mask = np.array(mask)

    # Create overlay
    overlay = image.copy()
    foreground = mask > 0

    # Highlight annotated object pixels
    overlay[foreground] = (
        0.5 * image[foreground] +
        0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    # Display original image
    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f"RGB Image: {image_id}")

    # Display GT mask
    axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[row, 1].set_title("Ground Truth Mask")

    # Display overlay
    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title("RGB + GT Overlay")

    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
plt.savefig("dataset_inspection.png", dpi=150)
plt.show()

print(f"Displayed {len(selected)} image-mask pairs.")
print("Saved visualization: dataset_inspection.png")
