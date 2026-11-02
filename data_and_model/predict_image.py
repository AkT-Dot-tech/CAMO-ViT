
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from transformers import SegformerForSemanticSegmentation


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_model.pth"

OUTPUT_DIR = ROOT / "outputs" / "standalone_inference"

IMAGE_SIZE = 384

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

def load_model():

    print("Loading SegFormer-B0...")

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        ignore_mismatched_sizes=True
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)   # type: ignore
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
    print(f"Device: {DEVICE}")

    return model


# --------------------------------------------------
# PREPROCESS IMAGE
# --------------------------------------------------

def preprocess_image(image_path):

    image = Image.open(image_path).convert("RGB")

    original_image = np.array(image)

    resized_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.array(resized_image).astype(np.float32) / 255.0

    normalized = (image_array - MEAN) / STD

    tensor = torch.from_numpy(normalized)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return original_image, tensor.to(DEVICE)


# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------

@torch.inference_mode()
def predict(model, image_tensor, original_shape):

    outputs = model(pixel_values=image_tensor)

    logits = F.interpolate(
        outputs.logits,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    probabilities = torch.softmax(logits, dim=1)

    foreground_probability = probabilities[0, 1]

    prediction = torch.argmax(logits, dim=1)[0]

    prediction = prediction.cpu().numpy().astype(np.uint8)
    probability = foreground_probability.cpu().numpy()

    original_height, original_width = original_shape[:2]

    # Restore prediction to original image dimensions.
    prediction_image = Image.fromarray(prediction * 255)

    prediction_image = prediction_image.resize(
        (original_width, original_height),
        Image.Resampling.NEAREST
    )

    prediction = np.array(prediction_image) > 127

    # Restore probability map to original dimensions.
    probability_image = Image.fromarray(
        (probability * 255).astype(np.uint8)
    )

    probability_image = probability_image.resize(
        (original_width, original_height),
        Image.Resampling.BILINEAR
    )

    probability = np.array(probability_image).astype(np.float32) / 255.0

    return prediction, probability


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

def save_results(original, prediction, probability, image_path):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = Path(image_path).stem

    mask_path = OUTPUT_DIR / f"{filename}_mask.png"
    overlay_path = OUTPUT_DIR / f"{filename}_overlay.png"
    heatmap_path = OUTPUT_DIR / f"{filename}_probability.png"
    comparison_path = OUTPUT_DIR / f"{filename}_comparison.png"

    # Binary mask
    mask = (prediction.astype(np.uint8) * 255)

    Image.fromarray(mask).save(mask_path)

    # Red overlay
    overlay = original.astype(np.float32).copy()

    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 0] = 255

    alpha = 0.45

    overlay[prediction] = (
        (1 - alpha) * overlay[prediction]
        + alpha * red_layer[prediction]
    )

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    Image.fromarray(overlay).save(overlay_path)

    # Probability heatmap
    plt.imsave(
        heatmap_path,
        probability,
        cmap="jet",
        vmin=0,
        vmax=1
    )

    # Combined comparison
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Predicted Mask")

    axes[2].imshow(overlay)
    axes[2].set_title("Prediction Overlay")

    heat = axes[3].imshow(probability, cmap="jet", vmin=0, vmax=1)
    axes[3].set_title("Foreground Probability")
    fig.colorbar(heat, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("\nInference completed!")
    print(f"Mask:        {mask_path}")
    print(f"Overlay:     {overlay_path}")
    print(f"Probability: {heatmap_path}")
    print(f"Comparison:  {comparison_path}")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="CAMO-ViT standalone image inference"
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to an RGB image"
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("\nCAMO-ViT Standalone Inference")
    print("--------------------------------")

    model = load_model()

    original, image_tensor = preprocess_image(image_path)

    print(f"Input image: {image_path}")
    print(f"Original dimensions: {original.shape[1]} x {original.shape[0]}")

    prediction, probability = predict(
        model,
        image_tensor,
        original.shape
    )

    save_results(
        original,
        prediction,
        probability,
        image_path
    )


if __name__ == "__main__":
    main()
