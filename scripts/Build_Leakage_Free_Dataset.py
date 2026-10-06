from pathlib import Path
from collections import defaultdict
import pandas as pd
import hashlib
import shutil
import yaml
import json


# =========================================================
# CONFIGURATION
# =========================================================

SOURCE = Path(
    r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_CLEAN"
)

DEST = Path(
    r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_LEAKAGE_FREE"
)

MANIFEST_FILE = SOURCE / "dataset_manifest.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper"
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def sha256_file(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def text_hash(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# =========================================================
# SAFETY
# =========================================================

if DEST.exists():
    raise RuntimeError(
        f"\nDestination already exists:\n{DEST}\n\n"
        "Delete or rename it manually before running again."
    )

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(MANIFEST_FILE)


# =========================================================
# LOAD MANIFEST
# =========================================================

df = pd.read_csv(MANIFEST_FILE)

required = {
    "filename",
    "source_group",
    "sha256",
    "split"
}

missing_columns = required - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing manifest columns: {missing_columns}"
    )

df["split"] = (
    df["split"]
    .astype(str)
    .str.strip()
    .str.lower()
    .replace({"valid": "val"})
)

if not df["split"].isin(
    ["train", "val", "test"]
).all():
    raise ValueError(
        "Manifest contains invalid split names."
    )

if df["filename"].duplicated().any():
    raise ValueError(
        "Manifest contains duplicate filenames."
    )

print("=" * 65)
print("LEAKAGE-FREE DATASET RECONSTRUCTION")
print("=" * 65)

print("\nManifest rows:", len(df))
print("\nManifest split distribution:")
print(df["split"].value_counts())


# =========================================================
# INDEX ALL CURRENT IMAGES
# =========================================================

print("\nIndexing current dataset images...")

image_index = defaultdict(list)

image_root = SOURCE / "images"

for path in image_root.rglob("*"):

    if (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
    ):
        image_index[path.name].append(path)

print(
    "Current image files:",
    sum(len(v) for v in image_index.values())
)

print(
    "Unique filenames:",
    len(image_index)
)


# =========================================================
# INDEX ALL CURRENT LABELS
# =========================================================

label_index = defaultdict(list)

label_root = SOURCE / "labels"

for path in label_root.rglob("*.txt"):

    if path.is_file():
        label_index[path.name].append(path)

print(
    "Current label files:",
    sum(len(v) for v in label_index.values())
)


# =========================================================
# CREATE DESTINATION STRUCTURE
# =========================================================

for split in ["train", "val", "test"]:

    (DEST / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (DEST / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================================================
# RECONSTRUCT USING SHA256 + MANIFEST SPLIT
# =========================================================

problems = []
records = []

for idx, row in df.iterrows():

    filename = row["filename"]
    expected_hash = str(row["sha256"]).lower()
    target_split = row["split"]

    candidates = image_index.get(
        filename,
        []
    )

    if not candidates:

        problems.append({
            "filename": filename,
            "problem": "Image not found"
        })

        continue

    # Find candidate whose bytes exactly match manifest SHA256
    matching_images = []

    for image_path in candidates:

        actual_hash = sha256_file(image_path)

        if actual_hash.lower() == expected_hash:
            matching_images.append(image_path)

    if len(matching_images) == 0:

        problems.append({
            "filename": filename,
            "problem":
                "No current image matches manifest SHA256"
        })

        continue

    # Both duplicated copies may be identical.
    # Choosing one is safe because SHA256 is identical.
    selected_image = matching_images[0]

    # -----------------------------------------------------
    # Find corresponding label
    # -----------------------------------------------------

    label_name = Path(filename).with_suffix(
        ".txt"
    ).name

    label_candidates = label_index.get(
        label_name,
        []
    )

    if not label_candidates:

        problems.append({
            "filename": filename,
            "problem": "Label not found"
        })

        continue

    # Check whether duplicated labels disagree
    label_hashes = defaultdict(list)

    for lp in label_candidates:
        label_hashes[text_hash(lp)].append(lp)

    if len(label_hashes) > 1:

        # First try label matching selected image relative path
        try:
            relative_image = selected_image.relative_to(
                SOURCE / "images"
            )

            expected_label = (
                SOURCE
                / "labels"
                / relative_image.with_suffix(".txt")
            )

        except Exception:
            expected_label = None

        if (
            expected_label is None
            or not expected_label.exists()
        ):

            problems.append({
                "filename": filename,
                "problem":
                    "Multiple labels have different content"
            })

            continue

        selected_label = expected_label

    else:
        selected_label = label_candidates[0]

    # -----------------------------------------------------
    # Copy to manifest-assigned split
    # -----------------------------------------------------

    destination_image = (
        DEST
        / "images"
        / target_split
        / filename
    )

    destination_label = (
        DEST
        / "labels"
        / target_split
        / label_name
    )

    shutil.copy2(
        selected_image,
        destination_image
    )

    shutil.copy2(
        selected_label,
        destination_label
    )

    records.append({
        "filename": filename,
        "source_group": row["source_group"],
        "sha256": expected_hash,
        "phash": row.get("phash", ""),
        "split": target_split,
        "image_path": str(destination_image),
        "label_path": str(destination_label)
    })

    if (
        (idx + 1) % 1000 == 0
        or idx + 1 == len(df)
    ):
        print(
            f"Processed {idx + 1}/{len(df)}"
        )


# =========================================================
# STOP IF ANY FILE COULD NOT BE VERIFIED
# =========================================================

if problems:

    problem_df = pd.DataFrame(problems)

    problem_file = (
        DEST / "reconstruction_problems.csv"
    )

    problem_df.to_csv(
        problem_file,
        index=False
    )

    raise RuntimeError(
        f"\nReconstruction found "
        f"{len(problems)} problems.\n"
        f"See:\n{problem_file}"
    )


# =========================================================
# SAVE NEW MANIFEST
# =========================================================

new_manifest = pd.DataFrame(records)

new_manifest.to_csv(
    DEST / "dataset_manifest.csv",
    index=False
)


# =========================================================
# CREATE data.yaml
# =========================================================

yaml_data = {
    "path": str(DEST).replace("\\", "/"),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 6,
    "names": CLASS_NAMES
}

with open(
    DEST / "data.yaml",
    "w",
    encoding="utf-8"
) as f:

    yaml.safe_dump(
        yaml_data,
        f,
        sort_keys=False,
        allow_unicode=True
    )


# =========================================================
# FINAL DATASET AUDIT
# =========================================================

print("\n" + "=" * 65)
print("FINAL DATASET AUDIT")
print("=" * 65)

audit = {}

all_hashes = defaultdict(set)

for split in ["train", "val", "test"]:

    img_dir = DEST / "images" / split
    lbl_dir = DEST / "labels" / split

    images = [
        p for p in img_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ]

    labels = list(
        lbl_dir.glob("*.txt")
    )

    missing_labels = 0

    for img in images:

        label = (
            lbl_dir
            / img.with_suffix(".txt").name
        )

        if not label.exists():
            missing_labels += 1

        h = sha256_file(img)

        all_hashes[h].add(split)

    audit[split] = {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": missing_labels
    }

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print(
        "Missing labels:",
        missing_labels
    )


# =========================================================
# EXACT DUPLICATE LEAKAGE
# =========================================================

train_val = 0
train_test = 0
val_test = 0

for splits in all_hashes.values():

    if "train" in splits and "val" in splits:
        train_val += 1

    if "train" in splits and "test" in splits:
        train_test += 1

    if "val" in splits and "test" in splits:
        val_test += 1


print("\nExact duplicate leakage:")

print(
    "Train ↔ Val:",
    train_val
)

print(
    "Train ↔ Test:",
    train_test
)

print(
    "Val ↔ Test:",
    val_test
)


# =========================================================
# SOURCE GROUP LEAKAGE
# =========================================================

group_split_count = (
    new_manifest
    .groupby("source_group")["split"]
    .nunique()
)

leaked_groups = group_split_count[
    group_split_count > 1
]

print(
    "\nSource-group leakage:",
    len(leaked_groups)
)


# =========================================================
# SPLIT DISTRIBUTION
# =========================================================

print("\nFinal split distribution:")

print(
    new_manifest["split"]
    .value_counts()
)


# =========================================================
# SAVE AUDIT REPORT
# =========================================================

audit_report = {
    "manifest_rows":
        len(new_manifest),

    "train_images":
        audit["train"]["images"],

    "val_images":
        audit["val"]["images"],

    "test_images":
        audit["test"]["images"],

    "train_val_exact_leakage":
        train_val,

    "train_test_exact_leakage":
        train_test,

    "val_test_exact_leakage":
        val_test,

    "source_group_leakage":
        int(len(leaked_groups))
}

with open(
    DEST / "leakage_audit.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        audit_report,
        f,
        indent=4
    )


print("\n" + "=" * 65)

if (
    train_val == 0
    and train_test == 0
    and val_test == 0
    and len(leaked_groups) == 0
):

    print(
        "SUCCESS: NO EXACT OR SOURCE-GROUP "
        "CROSS-SPLIT LEAKAGE DETECTED"
    )

else:

    print(
        "WARNING: LEAKAGE STILL EXISTS"
    )

print("=" * 65)

print(
    "\nNew dataset:",
    DEST
)

print(
    "New data.yaml:",
    DEST / "data.yaml"
)