from pathlib import Path
import hashlib
import csv
from collections import defaultdict

import cv2


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(r"C:\PCB_IEEE_ACCESS")
DATASET_ROOT = ROOT / "dataset" / "VOC_PCB_LEAKAGE_FREE"

IMAGE_DIRS = [
    DATASET_ROOT / "images" / "train",
    DATASET_ROOT / "images" / "val",
    DATASET_ROOT / "images" / "test",
]

LABEL_DIRS = [
    DATASET_ROOT / "labels" / "train",
    DATASET_ROOT / "labels" / "val",
    DATASET_ROOT / "labels" / "test",
]

OUTPUT_DIR = ROOT / "analysis" / "duplicate_label_conflicts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "duplicate_label_conflicts.csv"
REPORT_PATH = OUTPUT_DIR / "DUPLICATE_LABEL_CONFLICT_REPORT.txt"
VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_label_file(label_path: Path):
    if not label_path.exists():
        return []

    rows = []

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            rows.append(
                {
                    "raw": line,
                    "valid": False,
                }
            )
            continue

        try:
            cls_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            rows.append(
                {
                    "raw": line,
                    "valid": True,
                    "class_id": cls_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )

        except Exception:
            rows.append(
                {
                    "raw": line,
                    "valid": False,
                }
            )

    return rows


def normalized_label_signature(rows):
    valid_rows = []

    for row in rows:

        if not row.get("valid", False):
            valid_rows.append(
                ("INVALID", row["raw"])
            )
            continue

        valid_rows.append(
            (
                row["class_id"],
                round(row["x"], 8),
                round(row["y"], 8),
                round(row["w"], 8),
                round(row["h"], 8),
            )
        )

    return tuple(sorted(valid_rows, key=str))


def find_label_for_image(image_path: Path):
    split = image_path.parent.name

    label_path = (
        DATASET_ROOT
        / "labels"
        / split
        / f"{image_path.stem}.txt"
    )

    return label_path


def draw_labels(image_path: Path, rows, out_path: Path):
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    h_img, w_img = image.shape[:2]

    for row in rows:

        if not row.get("valid", False):
            continue

        cls_id = row["class_id"]

        xc = row["x"] * w_img
        yc = row["y"] * h_img
        bw = row["w"] * w_img
        bh = row["h"] * h_img

        x1 = int(round(xc - bw / 2))
        y1 = int(round(yc - bh / 2))
        x2 = int(round(xc + bw / 2))
        y2 = int(round(yc + bh / 2))

        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        if 0 <= cls_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[cls_id]
        else:
            class_name = f"class_{cls_id}"

        cv2.putText(
            image,
            class_name,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), image)

    return True


# ============================================================
# COLLECT IMAGES
# ============================================================

image_files = []

for image_dir in IMAGE_DIRS:

    for ext in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp",
    ]:

        image_files.extend(
            image_dir.glob(ext)
        )

print("=" * 80)
print("DUPLICATE LABEL CONFLICT AUDIT")
print("=" * 80)

print(
    f"\nTotal images found: {len(image_files)}"
)


# ============================================================
# HASH IMAGES
# ============================================================

hash_groups = defaultdict(list)

for i, image_path in enumerate(
    image_files,
    start=1,
):

    image_hash = sha256_file(image_path)

    hash_groups[image_hash].append(
        image_path
    )

    if i % 1000 == 0:
        print(
            f"Hashed {i}/{len(image_files)} images..."
        )


duplicate_groups = {
    h: paths
    for h, paths in hash_groups.items()
    if len(paths) > 1
}

print(
    f"\nExact duplicate image groups: "
    f"{len(duplicate_groups)}"
)


# ============================================================
# CHECK LABEL CONFLICTS
# ============================================================

conflict_groups = []

for image_hash, paths in duplicate_groups.items():

    signatures = []

    group_records = []

    for image_path in sorted(paths):

        label_path = find_label_for_image(
            image_path
        )

        rows = read_label_file(
            label_path
        )

        signature = normalized_label_signature(
            rows
        )

        signatures.append(signature)

        group_records.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "rows": rows,
                "signature": signature,
            }
        )

    unique_signatures = set(
        signatures
    )

    if len(unique_signatures) > 1:

        conflict_groups.append(
            {
                "sha256": image_hash,
                "records": group_records,
            }
        )


print(
    f"Duplicate groups with different labels: "
    f"{len(conflict_groups)}"
)


# ============================================================
# WRITE CSV + VISUALIZATIONS
# ============================================================

csv_rows = []

for group_index, group in enumerate(
    conflict_groups,
    start=1,
):

    group_id = f"conflict_{group_index:03d}"

    group_vis_dir = (
        VIS_DIR / group_id
    )

    group_vis_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for record_index, record in enumerate(
        group["records"],
        start=1,
    ):

        image_path = record["image_path"]
        label_path = record["label_path"]
        rows = record["rows"]

        valid_labels = [
            row
            for row in rows
            if row.get("valid", False)
        ]

        class_ids = [
            row["class_id"]
            for row in valid_labels
        ]

        class_names = []

        for cls_id in class_ids:

            if 0 <= cls_id < len(CLASS_NAMES):
                class_names.append(
                    CLASS_NAMES[cls_id]
                )
            else:
                class_names.append(
                    f"class_{cls_id}"
                )

        raw_labels = " | ".join(
            row["raw"]
            for row in rows
        )

        vis_name = (
            f"{record_index:02d}_"
            f"{image_path.stem}.jpg"
        )

        vis_path = (
            group_vis_dir
            / vis_name
        )

        draw_labels(
            image_path,
            rows,
            vis_path,
        )

        csv_rows.append(
            {
                "group_id": group_id,
                "sha256": group["sha256"],
                "filename": image_path.name,
                "split": image_path.parent.name,
                "image_path": str(image_path),
                "label_path": str(label_path),
                "label_count": len(valid_labels),
                "class_ids": ",".join(
                    map(str, class_ids)
                ),
                "class_names": ",".join(
                    class_names
                ),
                "raw_labels": raw_labels,
                "visualization": str(vis_path),
            }
        )


with CSV_PATH.open(
    "w",
    newline="",
    encoding="utf-8",
) as f:

    fieldnames = [
        "group_id",
        "sha256",
        "filename",
        "split",
        "image_path",
        "label_path",
        "label_count",
        "class_ids",
        "class_names",
        "raw_labels",
        "visualization",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(csv_rows)


# ============================================================
# WRITE REPORT
# ============================================================

report_lines = []

report_lines.append(
    "DUPLICATE LABEL CONFLICT AUDIT"
)

report_lines.append("=" * 80)

report_lines.append(
    f"Total images: {len(image_files)}"
)

report_lines.append(
    f"Exact duplicate image groups: "
    f"{len(duplicate_groups)}"
)

report_lines.append(
    f"Duplicate groups with different labels: "
    f"{len(conflict_groups)}"
)

report_lines.append("")

for group_index, group in enumerate(
    conflict_groups,
    start=1,
):

    group_id = f"conflict_{group_index:03d}"

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        f"{group_id}"
    )

    report_lines.append(
        f"SHA256: {group['sha256']}"
    )

    for record in group["records"]:

        report_lines.append(
            f"\nImage: "
            f"{record['image_path'].name}"
        )

        report_lines.append(
            f"Split: "
            f"{record['image_path'].parent.name}"
        )

        report_lines.append(
            f"Label file: "
            f"{record['label_path']}"
        )

        if not record["rows"]:
            report_lines.append(
                "Labels: EMPTY"
            )

        else:

            for row in record["rows"]:

                report_lines.append(
                    f"  {row['raw']}"
                )

    report_lines.append("")


REPORT_PATH.write_text(
    "\n".join(report_lines),
    encoding="utf-8",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

print(
    f"Conflict groups: "
    f"{len(conflict_groups)}"
)

print(
    f"\nCSV:\n{CSV_PATH}"
)

print(
    f"\nReport:\n{REPORT_PATH}"
)

print(
    f"\nVisualizations:\n{VIS_DIR}"
)

if len(conflict_groups) == 0:

    print(
        "\nStatus: PASS — "
        "no conflicting duplicate labels."
    )

else:

    print(
        "\nStatus: REVIEW REQUIRED"
    )

    print(
        "Do NOT automatically delete or merge "
        "these annotations."
    )

    print(
        "Manually inspect each conflict group."
    )