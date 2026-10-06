import multiprocessing
import torch
from ultralytics import YOLO
from pathlib import Path


def main():

    DATASET = Path(
        r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_EDGE_ENHANCED"
    )

    OUTPUT = Path(
        r"C:\PCB_IEEE_ACCESS\results"
    )

    assert torch.cuda.is_available(), "CUDA not available!"
    assert (DATASET / "data.yaml").exists(), "data.yaml not found!"

    print("=" * 60)
    print("EDGE-ENHANCED YOLO11n TRAINING")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Dataset:", DATASET)

    # Important:
    # Use the SAME fresh pretrained model as baseline
    model = YOLO("yolo11n.pt")

    results = model.train(

        data=str(DATASET / "data.yaml"),

        # Same as clean baseline
        epochs=100,
        imgsz=640,

        # Use explicit batch=16 because your baseline
        # AutoBatch actually fell back to batch 16
        batch=16,

        device=0,
        workers=4,

        cache="disk",
        amp=True,

        optimizer="auto",
        patience=20,
        close_mosaic=10,

        seed=42,

        # Same reproducibility setting as clean baseline
        deterministic=True,

        val=True,
        plots=True,

        save=True,
        save_period=10,

        project=str(OUTPUT),

        name="YOLO11n_EdgeEnhanced_LeakageFree",

        exist_ok=False
    )

    print("\nTraining completed!")
    print("Results saved to:", results.save_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()