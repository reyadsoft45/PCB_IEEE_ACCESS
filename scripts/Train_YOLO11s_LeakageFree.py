import multiprocessing
from pathlib import Path

import torch
from ultralytics import YOLO


def main():

    # =========================================================
    # PATHS
    # =========================================================

    ROOT = Path(r"C:\PCB_IEEE_ACCESS")

    DATA_YAML = (
        ROOT
        / "dataset"
        / "VOC_PCB_LEAKAGE_FREE"
        / "data.yaml"
    )

    RESULTS_DIR = (
        ROOT
        / "results"
    )

    # =========================================================
    # SAFETY CHECKS
    # =========================================================

    assert DATA_YAML.exists(), (
        f"Dataset YAML not found:\n{DATA_YAML}"
    )

    assert torch.cuda.is_available(), (
        "CUDA is not available!"
    )

    print("=" * 65)
    print("YOLO11s LEAKAGE-FREE TRAINING")
    print("=" * 65)

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Dataset:", DATA_YAML)

    # =========================================================
    # GPU SETTINGS
    # =========================================================

    torch.backends.cudnn.benchmark = False

    # =========================================================
    # LOAD FRESH PRETRAINED YOLO11s
    # =========================================================

    model = YOLO("yolo11s.pt")

    # =========================================================
    # TRAIN
    # =========================================================

    results = model.train(

        data=str(DATA_YAML),

        # -----------------------------------------------------
        # SAME CORE SETTINGS AS YOLO11n BASELINE
        # -----------------------------------------------------

        epochs=100,
        imgsz=640,

        # RTX 3060 12GB
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

        val=True,
        plots=True,

        save=True,
        save_period=10,

        project=str(RESULTS_DIR),

        name="YOLO11s_LeakageFree_640",

        exist_ok=False
    )

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)

    print("Results saved to:")
    print(results.save_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()