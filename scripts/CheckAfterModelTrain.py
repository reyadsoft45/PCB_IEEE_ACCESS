
import multiprocessing
import torch
from pathlib import Path
from ultralytics import YOLO


def main():

    # 1. Dataset and output
    DATASET = Path(
        r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_CLEAN"
    )

    OUTPUT = Path(
        r"C:\PCB_IEEE_ACCESS\results"
    )

    BEST_MODEL = (
        OUTPUT
        / "YOLO11_GPU_Optimized"
        / "weights"
        / "best.pt"
    )

    # 2. Verify files and CUDA
    assert BEST_MODEL.exists(), "best.pt not found"
    assert (DATASET / "data.yaml").exists()
    assert torch.cuda.is_available(), "CUDA not available"

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    # 3. Load baseline model
    model = YOLO(str(BEST_MODEL))

    # 4. Fine-tuning
    results = model.train(

        data=str(DATASET / "data.yaml"),

        # Training
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,

        # Data loading
        workers=4,
        cache="disk",

        # GPU
        amp=True,

        # Optimization
        optimizer="AdamW",
        lr0=0.0001,
        lrf=0.1,
        patience=10,

        # Augmentation
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.0,

        # Reproducibility
        seed=42,
        deterministic=True,

        # Evaluation
        val=True,
        plots=True,

        # Save results
        save=True,
        save_period=5,

        project=str(OUTPUT),
        name="YOLO11_FineTuned_Fixed",
        exist_ok=False
    )

    print("\nFine-tuning completed!")
    print("Results saved to:", results.save_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
