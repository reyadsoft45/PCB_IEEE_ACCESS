
import os
import torch
from ultralytics import YOLO
from pathlib import Path
import multiprocessing

def main():

    # Dataset configuration
    DATASET = Path(
        r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_CLEAN"
    )

    OUTPUT = Path(
        r"C:\PCB_IEEE_ACCESS\results"
    )

    # CUDA verification
    assert torch.cuda.is_available(), "CUDA not available!"

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    # GPU performance
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Load pretrained model
    model = YOLO("yolo11n.pt")

    # Start training
    results = model.train(

        data=str(DATASET / "data.yaml"),

        # Training
        epochs=100,
        imgsz=640,
        batch=0.70,
        device=0,

        # Data loading
        workers=4,
        cache="disk",


        # GPU optimization
        amp=True,

        # Training optimization
        optimizer="auto",
        patience=20,
        close_mosaic=10,

        # Reproducibility
        seed=42,
        deterministic=False,

        # Validation
        val=True,
        plots=True,

        # Save checkpoints
        save=True,
        save_period=10,

        project=str(OUTPUT),
        name="YOLO11_GPU_Optimized",
        exist_ok=False
    )

    print("Training completed!")
    print("Results:", results.save_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
