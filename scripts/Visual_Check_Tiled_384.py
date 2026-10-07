from pathlib import Path
import random
import cv2


ROOT = Path(r"C:\PCB_IEEE_ACCESS")

IMAGE_DIR = (
    ROOT
    / "dataset"
    / "VOC_PCB_TILED_TRAIN_384"
    / "images"
    / "train"
)

LABEL_DIR = (
    ROOT
    / "dataset"
    / "VOC_PCB_TILED_TRAIN_384"
    / "labels"
    / "train"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "tiled_train_384_visual_check"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}

NUM_SAMPLES = 30

random.seed(42)

image_files = sorted(
    IMAGE_DIR.glob("*.png")
)

if len(image_files) < NUM_SAMPLES:
    NUM_SAMPLES = len(image_files)

samples = random.sample(
    image_files,
    NUM_SAMPLES
)

for image_path in samples:

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue

    h, w = image.shape[:2]

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    if label_path.exists():

        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

        for line in lines:

            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(
                float(parts[0])
            )

            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])

            x1 = int(
                (xc - bw / 2)
                * w
            )

            y1 = int(
                (yc - bh / 2)
                * h
            )

            x2 = int(
                (xc + bw / 2)
                * w
            )

            y2 = int(
                (yc + bh / 2)
                * h
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            label = (
                CLASS_NAMES.get(
                    class_id,
                    str(class_id)
                )
            )

            cv2.putText(
                image,
                label,
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

    output_path = (
        OUTPUT_DIR
        / image_path.name
    )

    cv2.imwrite(
        str(output_path),
        image
    )

print("=" * 70)
print("VISUAL CHECK COMPLETE")
print("=" * 70)
print("Saved to:")
print(OUTPUT_DIR)