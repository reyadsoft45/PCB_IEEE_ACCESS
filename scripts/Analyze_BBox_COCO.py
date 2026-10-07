from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\PCB_IEEE_ACCESS")

DATASET = (
    ROOT
    / "dataset"
    / "VOC_PCB_LEAKAGE_FREE"
)

OUTPUT_DIR = (
    ROOT
    / "analysis"
    / "coco_bbox_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# IMPORTANT:
# Test is intentionally excluded.
SPLITS = ["train", "val"]

CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}


# ============================================================
# COCO SIZE DEFINITION
# ============================================================
#
# Small  : area < 32^2
# Medium : 32^2 <= area < 96^2
# Large  : area >= 96^2
#
# Pixel area is calculated in the ORIGINAL source image.
# ============================================================

SMALL_LIMIT = 32 ** 2       # 1024 px^2
MEDIUM_LIMIT = 96 ** 2      # 9216 px^2


def coco_size(area_px):

    if area_px < SMALL_LIMIT:
        return "small"

    elif area_px < MEDIUM_LIMIT:
        return "medium"

    else:
        return "large"


def find_image(image_dir, stem):

    extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    ]

    for ext in extensions:

        path = image_dir / f"{stem}{ext}"

        if path.exists():
            return path

    return None


# ============================================================
# READ DATA
# ============================================================

records = []

missing_images = []
invalid_labels = []

print("=" * 80)
print("COCO-STYLE PCB BOUNDING-BOX SIZE ANALYSIS")
print("DEVELOPMENT DATA ONLY: TRAIN + VAL")
print("=" * 80)


for split in SPLITS:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    if not label_dir.exists():
        raise FileNotFoundError(label_dir)

    labels = sorted(
        label_dir.glob("*.txt")
    )

    print(
        f"\nProcessing {split}: "
        f"{len(labels)} label files"
    )

    for label_path in labels:

        image_path = find_image(
            image_dir,
            label_path.stem
        )

        if image_path is None:

            missing_images.append(
                str(label_path)
            )

            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            missing_images.append(
                str(image_path)
            )

            continue

        img_h, img_w = image.shape[:2]

        image_area = (
            img_w * img_h
        )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

        for line_number, line in enumerate(
            lines,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:

                invalid_labels.append(
                    [
                        str(label_path),
                        line_number,
                        line,
                    ]
                )

                continue

            try:

                class_id = int(
                    float(parts[0])
                )

                x_center = float(parts[1])
                y_center = float(parts[2])

                width_norm = float(parts[3])
                height_norm = float(parts[4])

            except ValueError:

                invalid_labels.append(
                    [
                        str(label_path),
                        line_number,
                        line,
                    ]
                )

                continue

            if class_id not in CLASS_NAMES:

                invalid_labels.append(
                    [
                        str(label_path),
                        line_number,
                        line,
                    ]
                )

                continue

            # --------------------------------------------
            # YOLO normalized bbox -> pixel dimensions
            # --------------------------------------------

            bbox_width_px = (
                width_norm * img_w
            )

            bbox_height_px = (
                height_norm * img_h
            )

            bbox_area_px = (
                bbox_width_px
                * bbox_height_px
            )

            bbox_area_percent = (
                bbox_area_px
                / image_area
                * 100
            )

            category = coco_size(
                bbox_area_px
            )

            records.append({

                "split":
                    split,

                "image":
                    image_path.name,

                "class_id":
                    class_id,

                "class_name":
                    CLASS_NAMES[class_id],

                "image_width_px":
                    img_w,

                "image_height_px":
                    img_h,

                "bbox_width_px":
                    bbox_width_px,

                "bbox_height_px":
                    bbox_height_px,

                "bbox_area_px":
                    bbox_area_px,

                "bbox_area_percent":
                    bbox_area_percent,

                "coco_size":
                    category,
            })


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(
        "No bounding boxes found."
    )


# ============================================================
# SAVE ALL OBJECT RECORDS
# ============================================================

df.to_csv(
    OUTPUT_DIR
    / "coco_all_objects_train_val.csv",
    index=False
)


# ============================================================
# OVERALL DISTRIBUTION
# ============================================================

SIZE_ORDER = [
    "small",
    "medium",
    "large"
]

overall = (
    df["coco_size"]
    .value_counts()
    .reindex(
        SIZE_ORDER,
        fill_value=0
    )
)

overall_df = (
    overall
    .rename_axis("size")
    .reset_index(name="count")
)

overall_df["percentage"] = (
    overall_df["count"]
    / len(df)
    * 100
)

overall_df.to_csv(
    OUTPUT_DIR
    / "coco_overall_distribution.csv",
    index=False
)


# ============================================================
# PER CLASS DISTRIBUTION
# ============================================================

per_class = pd.crosstab(
    df["class_name"],
    df["coco_size"]
)

per_class = per_class.reindex(
    columns=SIZE_ORDER,
    fill_value=0
)

per_class["total"] = (
    per_class.sum(axis=1)
)

for category in SIZE_ORDER:

    per_class[
        f"{category}_percent"
    ] = (
        per_class[category]
        / per_class["total"]
        * 100
    )

per_class.to_csv(
    OUTPUT_DIR
    / "coco_per_class_distribution.csv"
)


# ============================================================
# SPLIT DISTRIBUTION
# ============================================================

per_split = pd.crosstab(
    df["split"],
    df["coco_size"]
)

per_split = per_split.reindex(
    columns=SIZE_ORDER,
    fill_value=0
)

per_split["total"] = (
    per_split.sum(axis=1)
)

per_split.to_csv(
    OUTPUT_DIR
    / "coco_split_distribution.csv"
)


# ============================================================
# PER CLASS NUMERIC STATISTICS
# ============================================================

stats = (
    df.groupby("class_name")
    .agg(

        object_count=(
            "bbox_area_px",
            "count"
        ),

        median_width_px=(
            "bbox_width_px",
            "median"
        ),

        median_height_px=(
            "bbox_height_px",
            "median"
        ),

        mean_width_px=(
            "bbox_width_px",
            "mean"
        ),

        mean_height_px=(
            "bbox_height_px",
            "mean"
        ),

        median_area_px=(
            "bbox_area_px",
            "median"
        ),

        mean_area_px=(
            "bbox_area_px",
            "mean"
        ),

        median_area_percent=(
            "bbox_area_percent",
            "median"
        ),

        mean_area_percent=(
            "bbox_area_percent",
            "mean"
        ),

    )
    .reset_index()
)

stats.to_csv(
    OUTPUT_DIR
    / "coco_per_class_statistics.csv",
    index=False
)


# ============================================================
# PLOT 1 - OVERALL DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    overall_df["size"],
    overall_df["count"]
)

plt.xlabel(
    "COCO object-size category"
)

plt.ylabel(
    "Number of bounding boxes"
)

plt.title(
    "PCB Defect Object-Size Distribution (Train + Val)"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "coco_overall_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2 - PER CLASS
# ============================================================

plot_data = per_class[
    [
        "small",
        "medium",
        "large",
    ]
]

ax = plot_data.plot(
    kind="bar",
    figsize=(12, 6)
)

ax.set_xlabel(
    "PCB defect class"
)

ax.set_ylabel(
    "Number of objects"
)

ax.set_title(
    "COCO Object-Size Distribution by Defect Class"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "coco_per_class_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3 - BBOX AREA HISTOGRAM
# ============================================================

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    df["bbox_area_px"],
    bins=60
)

plt.axvline(
    SMALL_LIMIT,
    linestyle="--"
)

plt.axvline(
    MEDIUM_LIMIT,
    linestyle="--"
)

plt.xlabel(
    "Bounding-box area (pixels²)"
)

plt.ylabel(
    "Number of objects"
)

plt.title(
    "PCB Bounding-Box Area Distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "coco_bbox_area_histogram.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE AUDIT ISSUES
# ============================================================

if missing_images:

    pd.DataFrame(
        {
            "missing_image":
                missing_images
        }
    ).to_csv(
        OUTPUT_DIR
        / "missing_images.csv",
        index=False
    )


if invalid_labels:

    pd.DataFrame(
        invalid_labels,
        columns=[
            "label_file",
            "line",
            "content",
        ]
    ).to_csv(
        OUTPUT_DIR
        / "invalid_labels.csv",
        index=False
    )


# ============================================================
# TEXT SUMMARY
# ============================================================

summary_path = (
    OUTPUT_DIR
    / "COCO_ANALYSIS_SUMMARY.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "COCO-STYLE PCB BOUNDING-BOX ANALYSIS\n"
    )

    f.write(
        "=====================================\n\n"
    )

    f.write(
        "Splits analyzed: TRAIN + VAL ONLY\n\n"
    )

    f.write(
        f"Total objects: {len(df)}\n\n"
    )

    f.write(
        "COCO thresholds:\n"
    )

    f.write(
        "Small  : area < 1024 px^2\n"
    )

    f.write(
        "Medium : 1024 <= area < 9216 px^2\n"
    )

    f.write(
        "Large  : area >= 9216 px^2\n\n"
    )

    f.write(
        "Overall distribution:\n"
    )

    f.write(
        overall_df.to_string(
            index=False
        )
    )

    f.write(
        "\n\nPer-class distribution:\n"
    )

    f.write(
        per_class.to_string()
    )

    f.write(
        "\n\nPer-class statistics:\n"
    )

    f.write(
        stats.round(3).to_string(
            index=False
        )
    )

    f.write(
        "\n\nAudit:\n"
    )

    f.write(
        f"Missing images: "
        f"{len(missing_images)}\n"
    )

    f.write(
        f"Invalid labels: "
        f"{len(invalid_labels)}\n"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("OVERALL COCO SIZE DISTRIBUTION")
print("=" * 80)

print(
    overall_df.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("PER CLASS")
print("=" * 80)

print(
    per_class.to_string()
)

print("\n" + "=" * 80)
print("PER CLASS STATISTICS")
print("=" * 80)

print(
    stats.round(3).to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("AUDIT")
print("=" * 80)

print(
    "Missing/unreadable images:",
    len(missing_images)
)

print(
    "Invalid annotation lines:",
    len(invalid_labels)
)

print("\nSaved to:")
print(OUTPUT_DIR)

print("\nCOCO analysis complete.")