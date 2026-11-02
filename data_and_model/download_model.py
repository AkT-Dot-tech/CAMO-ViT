from transformers import SegformerForSemanticSegmentation

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

print("Downloading SegFormer-B0...")

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME
)

print("\nModel downloaded successfully!")
print("Model:", MODEL_NAME)
print("Number of parameters:",
      sum(p.numel() for p in model.parameters()))