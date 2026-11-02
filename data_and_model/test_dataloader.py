
import torch
from torch.utils.data import DataLoader

from dataset import VIACDataset

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "VIAC" / "train_val_split.csv"

train_dataset = VIACDataset(
    csv_file=CSV_FILE,
    split="train",
    image_size=384
)

val_dataset = VIACDataset(
    csv_file=CSV_FILE,
    split="val",
    image_size=384
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

# Test one training batch
batch = next(iter(train_loader))

images = batch["pixel_values"]
masks = batch["labels"]

print("\nBATCH VERIFICATION")
print("Image shape:", images.shape)
print("Mask shape:", masks.shape)
print("Image dtype:", images.dtype)
print("Mask dtype:", masks.dtype)
print("Mask classes:", torch.unique(masks))

print("\nDataLoader working successfully!")
