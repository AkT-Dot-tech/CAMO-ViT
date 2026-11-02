
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation

from dataset import VIACDataset


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "VIAC" / "train_val_split.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

IMAGE_SIZE = 384
BATCH_SIZE = 1
EPOCHS = 30

LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# 2. LOSS FUNCTION: BCE + DICE
# --------------------------------------------------

bce_loss = nn.BCEWithLogitsLoss()


def calculate_loss(logits, masks):

    # Convert model output to original mask resolution
    logits = F.interpolate(
        logits,
        size=masks.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    # Convert masks to two-class one-hot format
    targets = F.one_hot(
        masks,
        num_classes=2
    ).permute(0, 3, 1, 2).float()

    # Binary cross-entropy
    bce = bce_loss(logits, targets)

    # Dice loss for foreground class
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    foreground_targets = targets[:, 1]

    intersection = (probabilities * foreground_targets).sum(
        dim=(1, 2)
    )

    denominator = probabilities.sum(dim=(1, 2)) + foreground_targets.sum(
        dim=(1, 2)
    )

    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)

    dice_loss = 1.0 - dice.mean()

    return bce + dice_loss


# --------------------------------------------------
# 3. VALIDATION METRICS
# --------------------------------------------------

def calculate_metrics(confusion):

    true_negative, false_positive = confusion[0]
    false_negative, true_positive = confusion[1]

    epsilon = 1e-7

    iou = true_positive / (
        true_positive + false_positive + false_negative + epsilon
    )

    dice = (2 * true_positive) / (
        2 * true_positive + false_positive + false_negative + epsilon
    )

    precision = true_positive / (
        true_positive + false_positive + epsilon
    )

    recall = true_positive / (
        true_positive + false_negative + epsilon
    )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall
    }


# --------------------------------------------------
# 4. LOAD DATASETS
# --------------------------------------------------

print("Loading datasets...")

train_dataset = VIACDataset(
    CSV_FILE,
    split="train",
    image_size=IMAGE_SIZE
)

val_dataset = VIACDataset(
    CSV_FILE,
    split="val",
    image_size=IMAGE_SIZE
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda")
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Device: {DEVICE}")


# --------------------------------------------------
# 5. LOAD MODEL
# --------------------------------------------------

print("\nLoading SegFormer-B0...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Freeze encoder
for parameter in model.segformer.parameters():
    parameter.requires_grad = False

# Train segmentation decoder
for parameter in model.decode_head.parameters():
    parameter.requires_grad = True

model = model.to(DEVICE) # type: ignore

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Trainable parameters: {trainable_parameters}")


# --------------------------------------------------
# 6. OPTIMIZER AND MIXED PRECISION
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=(DEVICE.type == "cuda")
)


# --------------------------------------------------
# 7. TRAINING AND VALIDATION
# --------------------------------------------------

best_val_dice = -1.0
epochs_without_improvement = 0

history = []

best_checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
history_path = OUTPUT_DIR / "training_history.csv"

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    # ---------------- TRAINING ----------------

    model.train()

    # Keep frozen encoder in evaluation mode
    model.segformer.eval()

    total_train_loss = 0.0

    for batch_index, batch in enumerate(train_loader):

        images = batch["pixel_values"].to(DEVICE)
        masks = batch["labels"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda")
        ):

            outputs = model(pixel_values=images)

            loss = calculate_loss(
                outputs.logits,
                masks
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_train_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_train_loss = total_train_loss / len(train_loader)

    # ---------------- VALIDATION ----------------

    model.eval()

    total_val_loss = 0.0

    confusion = torch.zeros(
        (2, 2),
        dtype=torch.int64,
        device=DEVICE
    )

    with torch.no_grad():

        for batch in val_loader:

            images = batch["pixel_values"].to(DEVICE)
            masks = batch["labels"].to(DEVICE)

            with torch.amp.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda")
            ):

                outputs = model(pixel_values=images)

                loss = calculate_loss(
                    outputs.logits,
                    masks
                )

            total_val_loss += loss.item()

            logits = F.interpolate(
                outputs.logits.float(),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            predictions = logits.argmax(dim=1)

            # Accumulate pixel-level confusion matrix
            encoded = masks * 2 + predictions

            confusion += torch.bincount(
                encoded.reshape(-1),
                minlength=4
            ).reshape(2, 2)

    average_val_loss = total_val_loss / len(val_loader)

    metrics = calculate_metrics(confusion)

    val_iou = metrics["iou"]
    val_dice = metrics["dice"]

    # ---------------- EPOCH RESULTS ----------------

    print("\n" + "=" * 55)
    print(f"EPOCH {epoch + 1}/{EPOCHS}")
    print("=" * 55)

    print(f"Training Loss:   {average_train_loss:.4f}")
    print(f"Validation Loss: {average_val_loss:.4f}")
    print(f"Validation IoU:  {val_iou:.4f}")
    print(f"Validation Dice: {val_dice:.4f}")
    print(f"Precision:       {metrics['precision']:.4f}")
    print(f"Recall:          {metrics['recall']:.4f}")

    # Store history
    history.append({
        "epoch": epoch + 1,
        "train_loss": average_train_loss,
        "val_loss": average_val_loss,
        "val_iou": val_iou,
        "val_dice": val_dice,
        "precision": metrics["precision"],
        "recall": metrics["recall"]
    })

    # Save training history after every epoch
    with open(history_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys()
        )

        writer.writeheader()
        writer.writerows(history)

    # ---------------- BEST CHECKPOINT ----------------

    if val_dice > best_val_dice:

        best_val_dice = val_dice
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "image_size": IMAGE_SIZE,
                "model_name": MODEL_NAME
            },
            best_checkpoint_path
        )

        print("\nBest model checkpoint saved!")

    else:

        epochs_without_improvement += 1

        print(
            f"\nNo improvement: "
            f"{epochs_without_improvement}/{EARLY_STOPPING_PATIENCE}"
        )

    # ---------------- EARLY STOPPING ----------------

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEarly stopping triggered.")
        break


# --------------------------------------------------
# 8. TRAINING COMPLETE
# --------------------------------------------------

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)

print(f"Best validation Dice: {best_val_dice:.4f}")
print(f"Best checkpoint: {best_checkpoint_path}")
print(f"Training history: {history_path}")

if DEVICE.type == "cuda":
    peak_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
    print(f"Peak GPU memory: {peak_memory:.2f} GB")

print("\nThe best validation model is saved.")
