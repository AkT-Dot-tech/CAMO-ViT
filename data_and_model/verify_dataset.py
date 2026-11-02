
from pathlib import Path
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "VIAC"

SPLITS = {
    "TRAIN": {
        "images": ROOT / "traindata" / "visible",
        "masks": ROOT / "traindata" / "GT",
        "expected": 1200
    },
    "TEST": {
        "images": ROOT / "testdata" / "visible",
        "masks": ROOT / "testdata" / "GT",
        "expected": 300
    }
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_files(folder):
    return {
        f.stem: f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    }


def verify_split(name, config):
    print(f"\n{'=' * 50}")
    print(f"VERIFYING {name} DATASET")
    print("=" * 50)

    image_dir = config["images"]
    mask_dir = config["masks"]

    if not image_dir.exists() or not mask_dir.exists():
        print("ERROR: Image or mask folder not found.")
        print("Expected:", image_dir, "and", mask_dir)
        return

    images = get_files(image_dir)
    masks = get_files(mask_dir)

    image_ids = set(images)
    mask_ids = set(masks)

    matching = image_ids & mask_ids
    missing_masks = image_ids - mask_ids
    missing_images = mask_ids - image_ids

    print("Images found:", len(images))
    print("Masks found:", len(masks))
    print("Matching pairs:", len(matching))
    print("Images without masks:", len(missing_masks))
    print("Masks without images:", len(missing_images))

    if missing_masks:
        print("Missing mask examples:", sorted(missing_masks)[:10])

    if missing_images:
        print("Missing image examples:", sorted(missing_images)[:10])

    dimension_errors = []
    invalid_files = []
    mask_details = {}

    for file_id in sorted(matching):
        try:
            with Image.open(images[file_id]) as img:
                image_size = img.size

            with Image.open(masks[file_id]) as mask:
                mask_size = mask.size
                mask_array = np.asarray(mask.convert("L"))

            if image_size != mask_size:
                dimension_errors.append(file_id)

            values = np.unique(mask_array)
            has_background = np.any(mask_array == 0)
            has_foreground = np.any(mask_array > 0)

            mask_details[file_id] = {
                "values": values,
                "background": has_background,
                "foreground": has_foreground
            }

        except Exception as error:
            invalid_files.append((file_id, str(error)))

    print("\nDimension mismatches:", len(dimension_errors))
    if dimension_errors:
        print("Examples:", dimension_errors[:10])

    print("Unreadable image/mask pairs:", len(invalid_files))
    if invalid_files:
        print("Examples:", invalid_files[:5])

    no_foreground = [
        k for k, v in mask_details.items()
        if not v["foreground"]
    ]

    no_background = [
        k for k, v in mask_details.items()
        if not v["background"]
    ]

    print("Masks without foreground:", len(no_foreground))
    print("Masks without background:", len(no_background))

    if mask_details:
        sample_id = next(iter(mask_details))
        print("\nSample mask:", sample_id)
        print("Sample pixel values:",
              mask_details[sample_id]["values"][:20])

    print("\nExpected image count:", config["expected"])
    print("Actual image count:", len(images))

    print("\nVERIFICATION SUMMARY:")
    if (
        len(images) == config["expected"]
        and len(masks) == config["expected"]
        and len(matching) == config["expected"]
        and not missing_masks
        and not missing_images
        and not dimension_errors
        and not invalid_files
        and not no_foreground
        and not no_background
    ):
        print("PASS: Basic dataset checks completed successfully.")
    else:
        print("CHECK REQUIRED: Some dataset issues were found.")

        
for split_name, split_config in SPLITS.items():
    verify_split(split_name, split_config)

print("\nDataset verification finished.")
