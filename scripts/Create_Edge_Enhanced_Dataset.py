from pathlib import Path
import shutil
import cv2
import numpy as np
import yaml

# =========================================================
# PATHS
# =========================================================

SOURCE = Path(
    r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_LEAKAGE_FREE"
)

DEST = Path(
    r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_EDGE_ENHANCED"
)

SPLITS = ["train", "val", "test"]

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

# =========================================================
# FPGA-TARGET LAPLACIAN SHARPENING
#
#       0 -1  0
#      -1  5 -1
#       0 -1  0
#
# out = 5*center - top - bottom - left - right
# =========================================================

kernel = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ],
    dtype=np.float32
)

# =========================================================
# SAFETY
# =========================================================

if DEST.exists():
    raise RuntimeError(
        f"Destination already exists:\n{DEST}\n"
        "Rename or delete it before running again."
    )

# =========================================================
# CREATE DATASET STRUCTURE
# =========================================================

for split in SPLITS:

    (DEST / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (DEST / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )

# =========================================================
# PROCESS IMAGES
# =========================================================

total = 0

for split in SPLITS:

    src_img_dir = SOURCE / "images" / split
    dst_img_dir = DEST / "images" / split

    src_lbl_dir = SOURCE / "labels" / split
    dst_lbl_dir = DEST / "labels" / split

    images = [
        p for p in src_img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]

    print(f"\nProcessing {split}: {len(images)} images")

    for idx, image_path in enumerate(images, 1):

        # Read image
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        # Apply Laplacian sharpening
        enhanced = cv2.filter2D(
            image,
            ddepth=-1,
            kernel=kernel,
            borderType=cv2.BORDER_REPLICATE
        )

        # Save with SAME filename
        output_path = (
            dst_img_dir
            / image_path.name
        )

        success = cv2.imwrite(
            str(output_path),
            enhanced
        )

        if not success:
            raise RuntimeError(
                f"Cannot save: {output_path}"
            )

        # Copy corresponding label
        label_path = (
            src_lbl_dir
            / image_path.with_suffix(".txt").name
        )

        if not label_path.exists():
            raise FileNotFoundError(
                label_path
            )

        shutil.copy2(
            label_path,
            dst_lbl_dir / label_path.name
        )

        total += 1

        if idx % 500 == 0:
            print(
                f"{split}: {idx}/{len(images)}"
            )

# =========================================================
# COPY MANIFEST
# =========================================================

manifest = SOURCE / "dataset_manifest.csv"

if manifest.exists():
    shutil.copy2(
        manifest,
        DEST / "dataset_manifest.csv"
    )

# =========================================================
# CREATE data.yaml
# =========================================================

data_yaml = {
    "path": str(DEST).replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 6,
    "names": {
        0: "missing_hole",
        1: "mouse_bite",
        2: "open_circuit",
        3: "short",
        4: "spur",
        5: "spurious_copper"
    }
}

with open(
    DEST / "data.yaml",
    "w",
    encoding="utf-8"
) as f:

    yaml.safe_dump(
        data_yaml,
        f,
        sort_keys=False,
        allow_unicode=True
    )

# =========================================================
# FINAL COUNT CHECK
# =========================================================

print("\n" + "=" * 60)
print("EDGE-ENHANCED DATASET COMPLETE")
print("=" * 60)

for split in SPLITS:

    image_count = len(
        list(
            (DEST / "images" / split)
            .glob("*")
        )
    )

    label_count = len(
        list(
            (DEST / "labels" / split)
            .glob("*.txt")
        )
    )

    print(
        f"{split}: "
        f"{image_count} images, "
        f"{label_count} labels"
    )

print("\nTotal processed:", total)

print("\nNew dataset:")
print(DEST)

print("\nNew data.yaml:")
print(DEST / "data.yaml")