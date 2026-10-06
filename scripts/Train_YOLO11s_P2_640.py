import multiprocessing
from pathlib import Path

import torch
from ultralytics import YOLO


def main():

    # ============================================================
    # PATHS
    # ============================================================

    ROOT = Path(r"C:\PCB_IEEE_ACCESS")

    DATA_YAML = (
        ROOT
        / "dataset"
        / "VOC_PCB_LEAKAGE_FREE"
        / "data.yaml"
    )

    MODEL_YAML = (
        ROOT
        / "models"
        / "yolo11s_p2.yaml"
    )

    RESULTS_DIR = ROOT / "results"


    # ============================================================
    # PRE-FLIGHT CHECK
    # ============================================================

    print("=" * 75)
    print("YOLO11s + P2 | PCB LEAKAGE-FREE TRAINING")
    print("=" * 75)

    assert DATA_YAML.exists(), (
        f"Dataset YAML not found:\n{DATA_YAML}"
    )

    assert MODEL_YAML.exists(), (
        f"Model YAML not found:\n{MODEL_YAML}"
    )

    assert torch.cuda.is_available(), (
        "CUDA is not available. Stop training."
    )

    print("\nPyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    props = torch.cuda.get_device_properties(0)

    print(
        "GPU memory:",
        round(props.total_memory / (1024 ** 3), 2),
        "GB"
    )

    print("\nDataset:")
    print(DATA_YAML)

    print("\nCustom model:")
    print(MODEL_YAML)


    # ============================================================
    # REPRODUCIBILITY
    # ============================================================

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


    # ============================================================
    # BUILD CUSTOM YOLO11s-P2 MODEL
    # ============================================================

    print("\nBuilding YOLO11s + P2...")

    model = YOLO(
        str(MODEL_YAML)
    )


    # ============================================================
    # TRANSFER YOLO11s PRETRAINED WEIGHTS
    # ============================================================

    print("\nLoading compatible YOLO11s pretrained weights...")

    model.load(
        "yolo11s.pt"
    )

    print("Compatible pretrained weights loaded.")


    # ============================================================
    # MODEL INFORMATION
    # ============================================================

    print("\nModel information:")

    model.info(
        verbose=False
    )


    # ============================================================
    # TRAIN
    # ============================================================

    results = model.train(

        # --------------------------------------------------------
        # DATA
        # --------------------------------------------------------

        data=str(DATA_YAML),

        # --------------------------------------------------------
        # CONTROLLED COMPARISON
        # Same settings as YOLO11s baseline
        # --------------------------------------------------------

        epochs=100,

        imgsz=640,

        batch=16,

        device=0,

        workers=4,

        cache="disk",

        # --------------------------------------------------------
        # PRECISION / PERFORMANCE
        # --------------------------------------------------------

        amp=True,

        # --------------------------------------------------------
        # OPTIMIZER
        # Keep comparable with previous experiment
        # --------------------------------------------------------

        optimizer="auto",

        # --------------------------------------------------------
        # EARLY STOPPING
        # --------------------------------------------------------

        patience=20,

        # --------------------------------------------------------
        # AUGMENTATION
        # Keep baseline defaults
        # --------------------------------------------------------

        close_mosaic=10,

        # --------------------------------------------------------
        # REPRODUCIBILITY
        # --------------------------------------------------------

        seed=42,

        deterministic=True,

        # --------------------------------------------------------
        # VALIDATION
        # --------------------------------------------------------

        val=True,

        plots=True,

        # --------------------------------------------------------
        # SAVING
        # --------------------------------------------------------

        save=True,

        save_period=10,

        project=str(RESULTS_DIR),

        name="YOLO11s_P2_LeakageFree_640",

        exist_ok=False
    )


    # ============================================================
    # COMPLETE
    # ============================================================

    print("\n" + "=" * 75)

    print("YOLO11s + P2 TRAINING COMPLETE")

    print("=" * 75)

    print("\nResults saved to:")

    print(results.save_dir)

    print(
        "\nIMPORTANT:\n"
        "Do NOT evaluate the held-out test set yet.\n"
        "First compare validation results against YOLO11s baseline."
    )


if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()