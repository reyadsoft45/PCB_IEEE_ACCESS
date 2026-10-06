import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


ROOT = Path(r"C:\PCB_IEEE_ACCESS")

DATA_YAML = (
    ROOT
    / "dataset"
    / "VOC_PCB_EDGE_ENHANCED"
    / "data.yaml"
)

BEST_MODEL = (
    ROOT
    / "results"
    / "YOLO11n_EdgeEnhanced_LeakageFree"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "YOLO11n_EdgeEnhanced_LeakageFree_TEST"
)

assert DATA_YAML.exists(), f"data.yaml not found: {DATA_YAML}"
assert BEST_MODEL.exists(), f"best.pt not found: {BEST_MODEL}"
assert torch.cuda.is_available(), "CUDA not available!"

print("=" * 60)
print("FINAL EDGE-ENHANCED TEST")
print("=" * 60)

print("GPU:", torch.cuda.get_device_name(0))
print("Model:", BEST_MODEL)
print("Dataset:", DATA_YAML)

model = YOLO(str(BEST_MODEL))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=16,
    device=0,
    workers=0,

    # Same evaluation settings as baseline
    conf=0.001,
    iou=0.7,

    plots=True,

    project=str(ROOT / "results"),
    name="YOLO11n_EdgeEnhanced_LeakageFree_TEST",
    exist_ok=True
)

precision = float(metrics.box.mp)
recall = float(metrics.box.mr)

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0 else 0
)

map50 = float(metrics.box.map50)
map5095 = float(metrics.box.map)

print("\n" + "=" * 60)
print("FINAL EDGE-ENHANCED TEST RESULTS")
print("=" * 60)

print(f"Precision   : {precision:.6f}")
print(f"Recall      : {recall:.6f}")
print(f"F1 Score    : {f1:.6f}")
print(f"mAP@0.5     : {map50:.6f}")
print(f"mAP@0.5:0.95: {map5095:.6f}")

rows = []

for i, cls_id in enumerate(metrics.box.ap_class_index):

    p, r, ap50, ap = metrics.box.class_result(i)

    class_f1 = (
        2 * p * r / (p + r)
        if (p + r) > 0 else 0
    )

    class_name = metrics.names[int(cls_id)]

    rows.append({
        "Class": class_name,
        "Precision": float(p),
        "Recall": float(r),
        "F1": float(class_f1),
        "mAP50": float(ap50),
        "mAP50-95": float(ap)
    })

df = pd.DataFrame(rows)

print("\n" + "=" * 60)
print("PER-CLASS EDGE-ENHANCED TEST RESULTS")
print("=" * 60)

print(df.to_string(index=False))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

overall_df = pd.DataFrame([{
    "Model": "YOLO11n Edge-Enhanced",
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "mAP50": map50,
    "mAP50-95": map5095
}])

overall_df.to_csv(
    OUTPUT_DIR / "overall_test_metrics.csv",
    index=False
)

df.to_csv(
    OUTPUT_DIR / "per_class_test_metrics.csv",
    index=False
)

print("\nSaved:")
print(OUTPUT_DIR / "overall_test_metrics.csv")
print(OUTPUT_DIR / "per_class_test_metrics.csv")

print("\nFINAL EDGE-ENHANCED TEST COMPLETE!")