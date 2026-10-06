from pathlib import Path
import csv
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\PCB_IEEE_ACCESS")
DATASET = ROOT / "dataset" / "VOC_PCB_LEAKAGE_FREE"
OUTPUT_DIR = ROOT / "analysis" / "bbox_size_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper",
}

SPLITS = ["train", "val", "test"]


# ============================================================
# SIZE CATEGORY
# ============================================================
#
# For our PCB dataset, we classify based on bbox area relative
# to the full image area.
#
# small  : < 1% of image area
# medium : 1% - 5%
# large  : > 5%
#
# These are dataset-analysis thresholds, NOT COCO AP_S/AP_M/AP_L.
# ============================================================

def get_size_category(area_ratio):
    if area_ratio < 0.01:
        return "small"
    elif area_ratio < 0.05:
        return "medium"
    else:
        return "large"


# ============================================================
# FIND MATCHING IMAGE
# ============================================================

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
        p = image_dir / f"{stem}{ext}"

        if p.exists():
            return p

    return None


# ============================================================
# READ DATASET
# ============================================================

records = []

missing_images = []
invalid_labels = []

print("=" * 80)
print("PCB BOUNDING-BOX SIZE ANALYSIS")
print("=" * 80)

for split in SPLITS:

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    print(f"\nProcessing split: {split}")

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found:\n{image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory not found:\n{label_dir}"
        )

    label_files = sorted(label_dir.glob("*.txt"))

    print("Label files:", len(label_files))

    for label_file in label_files:

        image_path = find_image(
            image_dir,
            label_file.stem
        )

        if image_path is None:

            missing_images.append(
                str(label_file)
            )

            continue

        img = cv2.imread(str(image_path))

        if img is None:

            missing_images.append(
                str(image_path)
            )

            continue

        img_h, img_w = img.shape[:2]

        image_area = img_w * img_h

        with open(
            label_file,
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
                    (
                        str(label_file),
                        line_number,
                        line
                    )
                )

                continue

            try:

                class_id = int(float(parts[0]))

                x_center = float(parts[1])
                y_center = float(parts[2])
                norm_w = float(parts[3])
                norm_h = float(parts[4])

            except ValueError:

                invalid_labels.append(
                    (
                        str(label_file),
                        line_number,
                        line
                    )
                )

                continue

            if class_id not in CLASS_NAMES:

                invalid_labels.append(
                    (
                        str(label_file),
                        line_number,
                        line
                    )
                )

                continue

            # ----------------------------------------------------
            # Convert normalized bbox dimensions to pixels
            # ----------------------------------------------------

            bbox_w_px = norm_w * img_w
            bbox_h_px = norm_h * img_h

            bbox_area_px = (
                bbox_w_px * bbox_h_px
            )

            area_ratio = (
                bbox_area_px / image_area
            )

            area_percent = (
                area_ratio * 100.0
            )

            size_category = get_size_category(
                area_ratio
            )

            aspect_ratio = (
                bbox_w_px / bbox_h_px
                if bbox_h_px > 0
                else np.nan
            )

            records.append({

                "split": split,

                "image_name": image_path.name,

                "label_name": label_file.name,

                "class_id": class_id,

                "class_name": CLASS_NAMES[class_id],

                "image_width_px": img_w,

                "image_height_px": img_h,

                "bbox_width_norm": norm_w,

                "bbox_height_norm": norm_h,

                "bbox_width_px": bbox_w_px,

                "bbox_height_px": bbox_h_px,

                "bbox_area_px": bbox_area_px,

                "bbox_area_ratio": area_ratio,

                "bbox_area_percent": area_percent,

                "aspect_ratio": aspect_ratio,

                "size_category": size_category,
            })


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError(
        "No bounding boxes were found."
    )

print("\n" + "=" * 80)
print("DATASET SUMMARY")
print("=" * 80)

print("Total bounding boxes:", len(df))

print("\nBoxes by split:")
print(
    df["split"]
    .value_counts()
    .sort_index()
)

print("\nBoxes by class:")
print(
    df["class_name"]
    .value_counts()
)


# ============================================================
# SAVE ALL BOX RECORDS
# ============================================================

all_boxes_csv = (
    OUTPUT_DIR
    / "all_bounding_boxes.csv"
)

df.to_csv(
    all_boxes_csv,
    index=False
)


# ============================================================
# OVERALL SIZE DISTRIBUTION
# ============================================================

size_order = [
    "small",
    "medium",
    "large"
]

overall_size = (
    df["size_category"]
    .value_counts()
    .reindex(
        size_order,
        fill_value=0
    )
)

overall_size_df = (
    overall_size
    .rename_axis("size_category")
    .reset_index(name="count")
)

overall_size_df["percentage"] = (
    overall_size_df["count"]
    / len(df)
    * 100
)

overall_size_csv = (
    OUTPUT_DIR
    / "overall_size_distribution.csv"
)

overall_size_df.to_csv(
    overall_size_csv,
    index=False
)


# ============================================================
# PER-CLASS SIZE DISTRIBUTION
# ============================================================

class_size = pd.crosstab(
    df["class_name"],
    df["size_category"]
)

class_size = class_size.reindex(
    columns=size_order,
    fill_value=0
)

class_size["total"] = (
    class_size.sum(axis=1)
)

for size in size_order:

    class_size[
        f"{size}_percent"
    ] = (
        class_size[size]
        / class_size["total"]
        * 100
    )

class_size_csv = (
    OUTPUT_DIR
    / "per_class_size_distribution.csv"
)

class_size.to_csv(
    class_size_csv
)


# ============================================================
# PER-CLASS STATISTICS
# ============================================================

stats = (
    df.groupby("class_name")
    .agg(

        object_count=(
            "bbox_area_percent",
            "count"
        ),

        mean_width_px=(
            "bbox_width_px",
            "mean"
        ),

        median_width_px=(
            "bbox_width_px",
            "median"
        ),

        mean_height_px=(
            "bbox_height_px",
            "mean"
        ),

        median_height_px=(
            "bbox_height_px",
            "median"
        ),

        mean_area_percent=(
            "bbox_area_percent",
            "mean"
        ),

        median_area_percent=(
            "bbox_area_percent",
            "median"
        ),

        min_area_percent=(
            "bbox_area_percent",
            "min"
        ),

        max_area_percent=(
            "bbox_area_percent",
            "max"
        ),

        mean_aspect_ratio=(
            "aspect_ratio",
            "mean"
        ),

    )
    .reset_index()
)

stats_csv = (
    OUTPUT_DIR
    / "per_class_bbox_statistics.csv"
)

stats.to_csv(
    stats_csv,
    index=False
)


# ============================================================
# SPLIT-WISE DISTRIBUTION
# ============================================================

split_size = pd.crosstab(
    df["split"],
    df["size_category"]
)

split_size = split_size.reindex(
    columns=size_order,
    fill_value=0
)

split_size["total"] = (
    split_size.sum(axis=1)
)

split_size_csv = (
    OUTPUT_DIR
    / "split_size_distribution.csv"
)

split_size.to_csv(
    split_size_csv
)


# ============================================================
# PLOT 1:
# Overall small / medium / large
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    overall_size_df["size_category"],
    overall_size_df["count"]
)

plt.xlabel(
    "Bounding-box size category"
)

plt.ylabel(
    "Number of objects"
)

plt.title(
    "Overall PCB Defect Bounding-Box Size Distribution"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "overall_size_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 2:
# Per-class small/medium/large
# ============================================================

plot_df = class_size[
    ["small", "medium", "large"]
]

ax = plot_df.plot(
    kind="bar",
    figsize=(12, 6)
)

ax.set_xlabel(
    "Defect class"
)

ax.set_ylabel(
    "Number of objects"
)

ax.set_title(
    "Bounding-Box Size Distribution by PCB Defect Class"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "per_class_size_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 3:
# Bounding-box area percentage distribution
# ============================================================

plt.figure(figsize=(9, 6))

plt.hist(
    df["bbox_area_percent"],
    bins=50
)

plt.xlabel(
    "Bounding-box area (% of image)"
)

plt.ylabel(
    "Number of objects"
)

plt.title(
    "Distribution of PCB Defect Bounding-Box Areas"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "bbox_area_histogram.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 4:
# Width vs height scatter
# ============================================================

plt.figure(figsize=(8, 7))

for class_name in CLASS_NAMES.values():

    subset = df[
        df["class_name"] == class_name
    ]

    plt.scatter(
        subset["bbox_width_px"],
        subset["bbox_height_px"],
        s=10,
        alpha=0.4,
        label=class_name
    )

plt.xlabel(
    "Bounding-box width (pixels)"
)

plt.ylabel(
    "Bounding-box height (pixels)"
)

plt.title(
    "PCB Defect Bounding-Box Width vs Height"
)

plt.legend(
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "bbox_width_vs_height.png",
    dpi=300
)

plt.close()


# ============================================================
# PLOT 5:
# Median bbox area per class
# ============================================================

median_area = (
    df.groupby("class_name")[
        "bbox_area_percent"
    ]
    .median()
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.bar(
    median_area.index,
    median_area.values
)

plt.xlabel(
    "Defect class"
)

plt.ylabel(
    "Median bbox area (% of image)"
)

plt.title(
    "Median Bounding-Box Area by PCB Defect Class"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    / "median_bbox_area_by_class.png",
    dpi=300
)

plt.close()


# ============================================================
# SAVE AUDIT ISSUES
# ============================================================

if missing_images:

    pd.DataFrame(
        {
            "missing_or_unreadable_image":
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
            "line_number",
            "content"
        ]
    ).to_csv(
        OUTPUT_DIR
        / "invalid_labels.csv",
        index=False
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("OVERALL SIZE DISTRIBUTION")
print("=" * 80)

print(
    overall_size_df.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("PER-CLASS SIZE DISTRIBUTION")
print("=" * 80)

print(
    class_size.to_string()
)


print("\n" + "=" * 80)
print("PER-CLASS BBOX STATISTICS")
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


print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

for file in sorted(
    OUTPUT_DIR.iterdir()
):
    print(file)


print("\nAnalysis complete.")