
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class VIACDataset(Dataset):

    def __init__(
        self,
        csv_file,
        split,
        image_size=384
    ):
        self.image_size = image_size
        self.samples = []

        with open(csv_file, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                if row["split"] == split:
                    self.samples.append({
                        "image": Path(row["image"]),
                        "mask": Path(row["mask"])
                    })

        if not self.samples:
            raise ValueError(f"No samples found for split: {split}")

        # SegFormer pretrained checkpoint normalization
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406]
        ).view(3, 1, 1)

        self.std = torch.tensor(
            [0.229, 0.224, 0.225]
        ).view(3, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        sample = self.samples[index]

        # Load RGB image
        image = Image.open(sample["image"]).convert("RGB")

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        image = np.asarray(image, dtype=np.float32) / 255.0

        # Convert HWC to CHW
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Normalize image
        image = (image - self.mean) / self.std

        # Load ground-truth mask
        mask = Image.open(sample["mask"]).convert("L")

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        mask = np.asarray(mask, dtype=np.uint8)

        # Convert 0/255 into binary 0/1 labels
        mask = (mask > 127).astype(np.int64)

        mask = torch.from_numpy(mask)

        return {
            "pixel_values": image,
            "labels": mask
        }
