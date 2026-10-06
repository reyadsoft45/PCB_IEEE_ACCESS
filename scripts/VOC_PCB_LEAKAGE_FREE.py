import multiprocessing
import torch
from ultralytics import YOLO
from pathlib import Path


def main():

    DATASET = Path(
        r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_LEAKAGE_FREE"
    )

    OUTPUT = Path(
        r"C:\PCB_IEEE_ACCESS\results"
    )

    assert torch.cuda.is_available(), "CUDA not available!"

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)

    torch.backends.cudnn.benchmark = True

    # IMPORTANT:
    # fresh pretrained YOLO11n
    model = YOLO("yolo11n.pt")

    results = model.train(

        data=str(DATASET / "data.yaml"),

        epochs=100,
        imgsz=640,

        # RTX 3060
        batch=0.70,
        device=0,

        workers=4,
        cache="disk",

        amp=True,

        optimizer="auto",
        patience=20,
        close_mosaic=10,

        seed=42,

        # Better reproducibility for paper
        deterministic=True,

        val=True,
        plots=True,

        save=True,
        save_period=10,

        project=str(OUTPUT),

        name="YOLO11n_LeakageFree_Baseline",

        exist_ok=False
    )

    print("\nTraining completed!")
    print("Results:", results.save_dir)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()