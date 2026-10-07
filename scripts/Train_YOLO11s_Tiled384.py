import multiprocessing
import platform
from pathlib import Path
from datetime import datetime

import torch
import ultralytics
from ultralytics import YOLO


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # PROJECT PATHS
    # ========================================================

    ROOT = Path(r"C:\PCB_IEEE_ACCESS")

    DATASET_ROOT = (
        ROOT
        / "dataset"
        / "VOC_PCB_TILED_TRAIN_384"
    )

    DATA_YAML = (
        DATASET_ROOT
        / "data_dev.yaml"
    )

    TILED_TRAIN_IMAGES = (
        DATASET_ROOT
        / "images"
        / "train"
    )

    TILED_TRAIN_LABELS = (
        DATASET_ROOT
        / "labels"
        / "train"
    )

    ORIGINAL_VAL_IMAGES = (
        ROOT
        / "dataset"
        / "VOC_PCB_LEAKAGE_FREE"
        / "images"
        / "val"
    )

    ORIGINAL_VAL_LABELS = (
        ROOT
        / "dataset"
        / "VOC_PCB_LEAKAGE_FREE"
        / "labels"
        / "val"
    )

    AUDIT_REPORT = (
        ROOT
        / "analysis"
        / "tiled_train_384_audit"
        / "TILED_DATASET_AUDIT_REPORT.txt"
    )

    RESULTS_DIR = (
        ROOT
        / "results"
    )

    RUN_NAME = (
        "YOLO11s_Tiled384_LeakageFree_640"
    )

    RUN_DIR = (
        RESULTS_DIR
        / RUN_NAME
    )

    # ========================================================
    # SAFETY CHECKS
    # ========================================================

    print("=" * 80)
    print("YOLO11s TILED-384 CONTROLLED TRAINING")
    print("=" * 80)

    required_paths = [
        DATA_YAML,
        TILED_TRAIN_IMAGES,
        TILED_TRAIN_LABELS,
        ORIGINAL_VAL_IMAGES,
        ORIGINAL_VAL_LABELS,
        AUDIT_REPORT,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired path not found:\n{path}"
            )

    # --------------------------------------------------------
    # Protect against accidental overwrite
    # --------------------------------------------------------

    if RUN_DIR.exists():

        raise RuntimeError(
            f"\nResult directory already exists:\n"
            f"{RUN_DIR}\n\n"
            "Do NOT overwrite a previous experiment.\n"
            "Rename the run or archive the old result first."
        )

    # ========================================================
    # VERIFY AUDIT PASSED
    # ========================================================

    audit_text = AUDIT_REPORT.read_text(
        encoding="utf-8"
    )

    if "Audit status: PASS" not in audit_text:

        raise RuntimeError(
            "\nTiled dataset audit did not report PASS.\n"
            "DO NOT TRAIN."
        )

    print("\nDataset audit: PASS")

    # ========================================================
    # VERIFY YAML DOES NOT CONTAIN TEST
    # ========================================================

    yaml_text = DATA_YAML.read_text(
        encoding="utf-8"
    )

    yaml_lower = yaml_text.lower()

    # Test should intentionally not be configured during model
    # development.
    active_test_lines = []

    for line in yaml_text.splitlines():

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("#"):
            continue

        if stripped.lower().startswith("test:"):

            active_test_lines.append(
                stripped
            )

    if active_test_lines:

        raise RuntimeError(
            "\nTEST SPLIT FOUND IN DEVELOPMENT YAML:\n"
            + "\n".join(active_test_lines)
            + "\n\nRemove test from the YAML before training."
        )

    print("Test split protection: PASS")

    # ========================================================
    # CUDA CHECK
    # ========================================================

    if not torch.cuda.is_available():

        raise RuntimeError(
            "\nCUDA is NOT available.\n"
            "Do not start this experiment on CPU."
        )

    gpu_name = torch.cuda.get_device_name(0)

    print("\n" + "=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)

    print(
        "Python:",
        platform.python_version()
    )

    print(
        "PyTorch:",
        torch.__version__
    )

    print(
        "Ultralytics:",
        ultralytics.__version__
    )

    print(
        "CUDA available:",
        torch.cuda.is_available()
    )

    print(
        "CUDA version:",
        torch.version.cuda
    )

    print(
        "GPU:",
        gpu_name
    )

    # ========================================================
    # DATASET COUNTS
    # ========================================================

    train_images = list(
        TILED_TRAIN_IMAGES.glob("*.png")
    )

    train_labels = list(
        TILED_TRAIN_LABELS.glob("*.txt")
    )

    val_images = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
    ]:

        val_images.extend(
            ORIGINAL_VAL_IMAGES.glob(
                extension
            )
        )

    val_labels = list(
        ORIGINAL_VAL_LABELS.glob("*.txt")
    )

    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)

    print(
        "Tiled train images:",
        len(train_images)
    )

    print(
        "Tiled train labels:",
        len(train_labels)
    )

    print(
        "Original validation images:",
        len(val_images)
    )

    print(
        "Original validation labels:",
        len(val_labels)
    )

    if len(train_images) != len(train_labels):

        raise RuntimeError(
            "Train image/label count mismatch."
        )

    if len(val_images) != len(val_labels):

        raise RuntimeError(
            "Validation image/label count mismatch."
        )

    if len(train_images) != 29868:

        print(
            "\nWARNING:"
            f"\nExpected approximately 29868 tiled train images,"
            f" but found {len(train_images)}."
        )

    if len(val_images) != 1598:

        print(
            "\nWARNING:"
            f"\nExpected 1598 validation images,"
            f" but found {len(val_images)}."
        )

    # ========================================================
    # EXPERIMENT DEFINITION
    # ========================================================

    print("\n" + "=" * 80)
    print("EXPERIMENT")
    print("=" * 80)

    print(
        "Model              : YOLO11s"
    )

    print(
        "Input size         : 640"
    )

    print(
        "Train data         : 384x384 tiled TRAIN"
    )

    print(
        "Validation data    : ORIGINAL leakage-free VAL"
    )

    print(
        "Test data          : NOT USED"
    )

    print(
        "Tile size          : 384"
    )

    print(
        "Overlap            : 25%"
    )

    print(
        "Epochs             : 100"
    )

    print(
        "Batch              : 16"
    )

    print(
        "Seed               : 42"
    )

    print(
        "Deterministic      : True"
    )

    print(
        "Baseline comparison: YOLO11s_LeakageFree_640"
    )

    print(
        "Baseline val mAP50-95: ~0.711"
    )

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    torch.manual_seed(42)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True

    # ========================================================
    # LOAD SAME PRETRAINED MODEL AS BASELINE
    # ========================================================

    print("\nLoading pretrained YOLO11s...")

    model = YOLO(
        "yolo11s.pt"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    start_time = datetime.now()

    print(
        "Start time:",
        start_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    results = model.train(

        # ====================================================
        # DATA
        # ====================================================

        data=str(DATA_YAML),

        # ====================================================
        # CONTROLLED SETTINGS
        # SAME AS YOLO11s BASELINE
        # ====================================================

        epochs=100,

        imgsz=640,

        batch=16,

        device=0,

        workers=4,

        cache="disk",

        amp=True,

        optimizer="auto",

        patience=20,

        close_mosaic=10,

        seed=42,

        deterministic=True,

        # ====================================================
        # VALIDATION
        # ====================================================

        val=True,

        plots=True,

        # ====================================================
        # SAVING
        # ====================================================

        save=True,

        save_period=10,

        project=str(
            RESULTS_DIR
        ),

        name=RUN_NAME,

        exist_ok=False,

        # ====================================================
        # IMPORTANT:
        # DO NOT ADD split="test"
        # ====================================================
    )

    end_time = datetime.now()

    duration = (
        end_time
        - start_time
    )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        "End time:",
        end_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print(
        "Duration:",
        duration
    )

    print(
        "\nResults saved to:"
    )

    print(
        results.save_dir
    )

    # ========================================================
    # FIND BEST MODEL
    # ========================================================

    best_model = (
        Path(results.save_dir)
        / "weights"
        / "best.pt"
    )

    if not best_model.exists():

        raise FileNotFoundError(
            f"\nTraining completed but best.pt "
            f"was not found:\n{best_model}"
        )

    print(
        "\nBest model:"
    )

    print(
        best_model
    )

    # ========================================================
    # SAVE EXPERIMENT METADATA
    # ========================================================

    metadata_file = (
        Path(results.save_dir)
        / "EXPERIMENT_METADATA.txt"
    )

    metadata_text = f"""
YOLO11s TILED-384 EXPERIMENT
========================================

PURPOSE
-------
Evaluate whether increasing effective PCB defect scale through
train-only 384x384 overlapping tiling improves localization
compared with the original YOLO11s @640 baseline.

DATA
----
Training:
{TILED_TRAIN_IMAGES}

Validation:
{ORIGINAL_VAL_IMAGES}

Test:
NOT USED

Tiled train images:
{len(train_images)}

Original validation images:
{len(val_images)}

TILING
------
Tile size: 384x384
Nominal overlap: 25%
Nominal stride: 288
Minimum retained bbox fraction: 50%

MODEL
-----
YOLO11s pretrained weights

TRAINING
--------
Epochs: 100
Input size: 640
Batch: 16
Workers: 4
Cache: disk
AMP: True
Optimizer: auto
Patience: 20
Close mosaic: 10
Seed: 42
Deterministic: True

SYSTEM
------
Python: {platform.python_version()}
PyTorch: {torch.__version__}
Ultralytics: {ultralytics.__version__}
CUDA: {torch.version.cuda}
GPU: {gpu_name}

TIMING
------
Start: {start_time}
End: {end_time}
Duration: {duration}

CONTROL BASELINE
----------------
YOLO11s @640
Original leakage-free train
Original leakage-free validation
Validation mAP50-95 approximately 0.711

IMPORTANT
---------
The held-out test set was NOT used during this experiment.

INTERPRETATION RULE
-------------------
Model selection must be based only on validation performance.
The main comparison is against the original YOLO11s validation
mAP50-95, with special attention to open_circuit localization.
"""

    metadata_file.write_text(
        metadata_text.strip() + "\n",
        encoding="utf-8"
    )

    print(
        "\nMetadata saved:"
    )

    print(
        metadata_file
    )

    # ========================================================
    # FINAL VALIDATION OF BEST MODEL
    # ========================================================

    print("\n" + "=" * 80)
    print("VALIDATING BEST.PT ON ORIGINAL VAL")
    print("=" * 80)

    best = YOLO(
        str(best_model)
    )

    metrics = best.val(

        data=str(DATA_YAML),

        split="val",

        imgsz=640,

        batch=16,

        device=0,

        workers=0,

        conf=0.001,

        iou=0.7,

        plots=True,

        project=str(
            RESULTS_DIR
        ),

        name=(
            RUN_NAME
            + "_FINAL_VAL"
        ),

        exist_ok=False,
    )

    # ========================================================
    # METRIC SUMMARY
    # ========================================================

    mp = float(
        metrics.box.mp
    )

    mr = float(
        metrics.box.mr
    )

    map50 = float(
        metrics.box.map50
    )

    map5095 = float(
        metrics.box.map
    )

    f1 = (
        2 * mp * mr
        / (mp + mr)
        if (mp + mr) > 0
        else 0
    )

    print("\n" + "=" * 80)
    print("FINAL VALIDATION RESULTS")
    print("=" * 80)

    print(
        f"Precision       : {mp:.6f}"
    )

    print(
        f"Recall          : {mr:.6f}"
    )

    print(
        f"F1              : {f1:.6f}"
    )

    print(
        f"mAP@0.50        : {map50:.6f}"
    )

    print(
        f"mAP@0.50:0.95   : {map5095:.6f}"
    )

    # ========================================================
    # PER CLASS AP50-95
    # ========================================================

    print("\n" + "=" * 80)
    print("PER-CLASS AP@0.50:0.95")
    print("=" * 80)

    class_names = best.names

    per_class_maps = (
        metrics.box.maps
    )

    class_lines = []

    for class_id, ap in enumerate(
        per_class_maps
    ):

        class_name = class_names[
            class_id
        ]

        ap_value = float(ap)

        print(
            f"{class_id}: "
            f"{class_name:<18} "
            f"{ap_value:.6f}"
        )

        class_lines.append(
            f"{class_id},"
            f"{class_name},"
            f"{ap_value:.6f}"
        )

    # ========================================================
    # SAVE SUMMARY FILE
    # ========================================================

    summary_file = (
        Path(results.save_dir)
        / "FINAL_VALIDATION_SUMMARY.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "YOLO11s TILED-384 FINAL VALIDATION\n"
        )

        f.write(
            "==================================\n\n"
        )

        f.write(
            "Validation dataset: "
            "ORIGINAL leakage-free validation\n"
        )

        f.write(
            "Test dataset: NOT USED\n\n"
        )

        f.write(
            f"Precision: {mp:.6f}\n"
        )

        f.write(
            f"Recall: {mr:.6f}\n"
        )

        f.write(
            f"F1: {f1:.6f}\n"
        )

        f.write(
            f"mAP50: {map50:.6f}\n"
        )

        f.write(
            f"mAP50-95: {map5095:.6f}\n\n"
        )

        f.write(
            "Per-class AP50-95\n"
        )

        f.write(
            "class_id,class_name,AP50-95\n"
        )

        for line in class_lines:

            f.write(
                line + "\n"
            )

    print(
        "\nSummary saved:"
    )

    print(
        summary_file
    )

    # ========================================================
    # BASELINE COMPARISON
    # ========================================================

    BASELINE_MAP5095 = 0.711

    delta = (
        map5095
        - BASELINE_MAP5095
    )

    print("\n" + "=" * 80)
    print("CONTROLLED COMPARISON")
    print("=" * 80)

    print(
        f"YOLO11s original baseline : "
        f"{BASELINE_MAP5095:.3f}"
    )

    print(
        f"YOLO11s tiled-384         : "
        f"{map5095:.3f}"
    )

    print(
        f"Absolute delta            : "
        f"{delta:+.3f}"
    )

    print(
        f"Percentage-point change   : "
        f"{delta * 100:+.2f} pp"
    )

    # ========================================================
    # DECISION MESSAGE
    # ========================================================

    print("\n" + "=" * 80)
    print("EXPERIMENT DECISION")
    print("=" * 80)

    if delta >= 0.01:

        print(
            "PROMISING RESULT."
        )

        print(
            "Tiling improved validation "
            "mAP50-95 by at least 1 percentage point."
        )

        print(
            "Next: inspect per-class AP and "
            "repeat key experiment with another seed."
        )

    elif delta > 0:

        print(
            "SMALL IMPROVEMENT."
        )

        print(
            "Do NOT claim superiority yet."
        )

        print(
            "The gain may be within run-to-run variation."
        )

    else:

        print(
            "NO IMPROVEMENT."
        )

        print(
            "Do NOT proceed to test-set evaluation."
        )

        print(
            "Keep this as a negative ablation result."
        )

    print("\nTEST SET REMAINS PROTECTED.")


# ============================================================
# WINDOWS ENTRY POINT
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()