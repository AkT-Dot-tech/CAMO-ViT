
import torch
from transformers import SegformerForSemanticSegmentation

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# Select GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading pretrained SegFormer-B0...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    ignore_mismatched_sizes=True
)

# Move model to GPU
model.to(device) # type: ignore
model.eval()

print("Model loaded on:", device)

# Simulate one RGB image
dummy_input = torch.randn(
    1, 3, 384, 384,
    device=device
)

with torch.no_grad():
    output = model(pixel_values=dummy_input)

print("Input:", dummy_input.shape)
print("Output:", output.logits.shape)

print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

print("SegFormer GPU inference successful!")
