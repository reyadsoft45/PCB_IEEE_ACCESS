from pathlib import Path
import torch
from ultralytics import YOLO


MODEL_YAML = Path(
    r"C:\PCB_IEEE_ACCESS\models\yolo11s_p2.yaml"
)

print("=" * 70)
print("YOLO11s + P2 ARCHITECTURE CHECK")
print("=" * 70)

print("\nPyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("\nModel YAML:")
print(MODEL_YAML)

assert MODEL_YAML.exists(), f"YAML not found: {MODEL_YAML}"

# Build custom architecture
model = YOLO(str(MODEL_YAML))

print("\nCustom model created successfully.")

# Display architecture information
model.info(verbose=True)

print("\n" + "=" * 70)
print("ARCHITECTURE CHECK SUCCESS")
print("=" * 70)