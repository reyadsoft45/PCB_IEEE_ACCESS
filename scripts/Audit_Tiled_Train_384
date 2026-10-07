from pathlib import Path
from collections import Counter
import cv2
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\PCB_IEEE_ACCESS")

DATASET = (
    ROOT
    / "dataset"
    / "VOC_PCB_TILED_TRAIN_384"
)

IMAGE_DIR = DATASET / "images" / "train"
LABEL_DIR = DATASET / "labels" / "train"

MANIFEST = DATASET / "tile_manifest.csv"

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "tiled_train_384_audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIG
# ============================================================

EXPECTED_SIZE = 384

CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}


# ============================================================
# PRE-CHECK
# ============================================================

print("=" * 80)
print("TILED TRAIN 384 DATASET AUDIT")
print("=" * 80)

assert IMAGE_DIR.exists(), f"Missing: {IMAGE_DIR}"
assert LABEL_DIR.exists(), f"Missing: {LABEL_DIR}"
assert MANIFEST.exists(), f"Missing: {MANIFEST}"


image_files = sorted(
    list(IMAGE_DIR.glob("*.png"))
)

label_files = sorted(
    list(LABEL_DIR.glob("*.txt"))
)


print("\nImages:", len(image_files))
print("Labels:", len(label_files))


# ============================================================
# IMAGE / LABEL MATCHING
# ============================================================

image_stems = {
    p.stem for p in image_files
}

label_stems = {
    p.stem for p in label_files
}

images_without_labels = sorted(
    image_stems - label_stems
)

labels_without_images = sorted(
    label_stems - image_stems
)


# ============================================================
# IMAGE SIZE CHECK
# ============================================================

wrong_size_images = []
unreadable_images = []

for image_path in image_files:

    img = cv2.imread(
        str(image_path)
    )

    if img is None:

        unreadable_images.append(
            image_path.name
        )

        continue

    h, w = img.shape[:2]

    if (
        h != EXPECTED_SIZE
        or w != EXPECTED_SIZE
    ):

        wrong_size_images.append(
            {
                "image": image_path.name,
                "width": w,
                "height": h,
            }
        )


# ============================================================
# LABEL CHECK
# ============================================================

invalid_lines = []

class_counts = Counter()

total_objects = 0
background_tiles = 0
positive_tiles = 0

bbox_widths = []
bbox_heights = []

for label_path in label_files:

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            x.strip()
            for x in f.readlines()
            if x.strip()
        ]

    if len(lines) == 0:

        background_tiles += 1
        continue

    positive_tiles += 1

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        parts = line.split()

        if len(parts) != 5:

            invalid_lines.append({
                "file": label_path.name,
                "line_number": line_number,
                "reason": "Expected 5 columns",
                "content": line,
            })

            continue

        try:

            class_id = int(
                float(parts[0])
            )

            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:

            invalid_lines.append({
                "file": label_path.name,
                "line_number": line_number,
                "reason": "Non-numeric value",
                "content": line,
            })

            continue

        # ----------------------------------------------------
        # Class ID check
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:

            invalid_lines.append({
                "file": label_path.name,
                "line_number": line_number,
                "reason": f"Invalid class {class_id}",
                "content": line,
            })

            continue

        # ----------------------------------------------------
        # YOLO normalized coordinate checks
        # ----------------------------------------------------

        if not (
            0 <= xc <= 1
            and 0 <= yc <= 1
            and 0 < w <= 1
            and 0 < h <= 1
        ):

            invalid_lines.append({
                "file": label_path.name,
                "line_number": line_number,
                "reason": "Coordinates outside valid YOLO range",
                "content": line,
            })

            continue

        # ----------------------------------------------------
        # Check actual box boundaries
        # ----------------------------------------------------

        x1 = xc - w / 2
        y1 = yc - h / 2

        x2 = xc + w / 2
        y2 = yc + h / 2

        tolerance = 1e-6

        if (
            x1 < -tolerance
            or y1 < -tolerance
            or x2 > 1 + tolerance
            or y2 > 1 + tolerance
        ):

            invalid_lines.append({
                "file": label_path.name,
                "line_number": line_number,
                "reason": "Bounding box exceeds tile boundary",
                "content": line,
            })

            continue

        total_objects += 1

        class_counts[class_id] += 1

        bbox_widths.append(
            w * EXPECTED_SIZE
        )

        bbox_heights.append(
            h * EXPECTED_SIZE
        )


# ============================================================
# MANIFEST AUDIT
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

required_columns = [
    "tile_filename",
    "source_image",
    "tile_x",
    "tile_y",
    "tile_size",
    "objects_in_tile",
    "is_background_tile",
]

missing_manifest_columns = [
    col
    for col in required_columns
    if col not in manifest.columns
]


manifest_duplicate_tiles = (
    manifest["tile_filename"]
    .duplicated()
    .sum()
    if "tile_filename" in manifest.columns
    else -1
)


# ============================================================
# SOURCE TRACEABILITY
# ============================================================

unique_source_images = (
    manifest["source_image"].nunique()
    if "source_image" in manifest.columns
    else -1
)


# ============================================================
# CLASS SUMMARY
# ============================================================

class_rows = []

for class_id, class_name in CLASS_NAMES.items():

    count = class_counts[class_id]

    class_rows.append({
        "class_id": class_id,
        "class_name": class_name,
        "object_count": count,
        "percentage": (
            count / total_objects * 100
            if total_objects > 0
            else 0
        ),
    })


class_df = pd.DataFrame(
    class_rows
)

class_df.to_csv(
    OUTPUT_DIR
    / "tiled_class_distribution.csv",
    index=False
)


# ============================================================
# SAVE AUDIT ISSUES
# ============================================================

if invalid_lines:

    pd.DataFrame(
        invalid_lines
    ).to_csv(
        OUTPUT_DIR
        / "invalid_labels.csv",
        index=False
    )


if wrong_size_images:

    pd.DataFrame(
        wrong_size_images
    ).to_csv(
        OUTPUT_DIR
        / "wrong_size_images.csv",
        index=False
    )


if images_without_labels:

    pd.DataFrame({
        "image_without_label":
            images_without_labels
    }).to_csv(
        OUTPUT_DIR
        / "images_without_labels.csv",
        index=False
    )


if labels_without_images:

    pd.DataFrame({
        "label_without_image":
            labels_without_images
    }).to_csv(
        OUTPUT_DIR
        / "labels_without_images.csv",
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

background_percent = (
    background_tiles
    / len(label_files)
    * 100
    if label_files
    else 0
)

positive_percent = (
    positive_tiles
    / len(label_files)
    * 100
    if label_files
    else 0
)


mean_width = (
    sum(bbox_widths)
    / len(bbox_widths)
    if bbox_widths
    else 0
)

mean_height = (
    sum(bbox_heights)
    / len(bbox_heights)
    if bbox_heights
    else 0
)


PASS = (
    len(images_without_labels) == 0
    and len(labels_without_images) == 0
    and len(unreadable_images) == 0
    and len(wrong_size_images) == 0
    and len(invalid_lines) == 0
    and len(missing_manifest_columns) == 0
    and manifest_duplicate_tiles == 0
)


print("\n" + "=" * 80)
print("AUDIT RESULTS")
print("=" * 80)

print(
    f"Image files              : {len(image_files)}"
)

print(
    f"Label files              : {len(label_files)}"
)

print(
    f"Positive tiles           : {positive_tiles}"
)

print(
    f"Background tiles         : {background_tiles}"
)

print(
    f"Positive tile %          : {positive_percent:.2f}%"
)

print(
    f"Background tile %        : {background_percent:.2f}%"
)

print(
    f"Total retained objects   : {total_objects}"
)

print(
    f"Unique source images     : {unique_source_images}"
)

print(
    f"Images without labels    : {len(images_without_labels)}"
)

print(
    f"Labels without images    : {len(labels_without_images)}"
)

print(
    f"Unreadable images        : {len(unreadable_images)}"
)

print(
    f"Wrong-size images        : {len(wrong_size_images)}"
)

print(
    f"Invalid label lines      : {len(invalid_lines)}"
)

print(
    f"Manifest duplicate tiles : {manifest_duplicate_tiles}"
)

print(
    f"Mean bbox width (px)     : {mean_width:.2f}"
)

print(
    f"Mean bbox height (px)    : {mean_height:.2f}"
)


print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

print(
    class_df.to_string(
        index=False
    )
)


print("\n" + "=" * 80)

if PASS:

    print(
        "AUDIT STATUS: PASS"
    )

    print(
        "TILED DATASET IS STRUCTURALLY READY "
        "FOR MODEL TRAINING."
    )

else:

    print(
        "AUDIT STATUS: FAIL"
    )

    print(
        "DO NOT START TRAINING."
    )


print("=" * 80)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_file = (
    OUTPUT_DIR
    / "TILED_DATASET_AUDIT_REPORT.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "TILED TRAIN 384 DATASET AUDIT\n"
    )

    f.write(
        "=============================\n\n"
    )

    f.write(
        f"Audit status: "
        f"{'PASS' if PASS else 'FAIL'}\n\n"
    )

    f.write(
        f"Images: {len(image_files)}\n"
    )

    f.write(
        f"Labels: {len(label_files)}\n"
    )

    f.write(
        f"Positive tiles: {positive_tiles}\n"
    )

    f.write(
        f"Background tiles: {background_tiles}\n"
    )

    f.write(
        f"Background percentage: "
        f"{background_percent:.2f}%\n"
    )

    f.write(
        f"Objects: {total_objects}\n"
    )

    f.write(
        f"Unique sources: "
        f"{unique_source_images}\n"
    )

    f.write(
        f"Invalid labels: "
        f"{len(invalid_lines)}\n"
    )

    f.write(
        f"Wrong image sizes: "
        f"{len(wrong_size_images)}\n"
    )

    f.write(
        f"Image-label mismatches: "
        f"{len(images_without_labels) + len(labels_without_images)}\n"
    )

    f.write(
        f"Duplicate manifest tiles: "
        f"{manifest_duplicate_tiles}\n\n"
    )

    f.write(
        "Class distribution:\n"
    )

    f.write(
        class_df.to_string(
            index=False
        )
    )


print("\nAudit report saved:")
print(report_file)