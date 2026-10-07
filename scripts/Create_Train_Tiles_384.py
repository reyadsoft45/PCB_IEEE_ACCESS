from pathlib import Path
import shutil
import cv2
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\PCB_IEEE_ACCESS")

SOURCE_DATASET = (
    ROOT
    / "dataset"
    / "VOC_PCB_LEAKAGE_FREE"
)

OUTPUT_DATASET = (
    ROOT
    / "dataset"
    / "VOC_PCB_TILED_TRAIN_384"
)

SOURCE_IMAGE_DIR = (
    SOURCE_DATASET
    / "images"
    / "train"
)

SOURCE_LABEL_DIR = (
    SOURCE_DATASET
    / "labels"
    / "train"
)

OUTPUT_IMAGE_DIR = (
    OUTPUT_DATASET
    / "images"
    / "train"
)

OUTPUT_LABEL_DIR = (
    OUTPUT_DATASET
    / "labels"
    / "train"
)

MANIFEST_PATH = (
    OUTPUT_DATASET
    / "tile_manifest.csv"
)

YAML_PATH = (
    OUTPUT_DATASET
    / "data_dev.yaml"
)


# ============================================================
# PARAMETERS
# ============================================================

TILE_SIZE = 384

# 25% nominal overlap
OVERLAP = 0.25

STRIDE = int(
    TILE_SIZE
    * (1 - OVERLAP)
)

# Keep clipped object only when at least
# 50% of its original bounding-box area remains.
MIN_RETAINED_AREA = 0.50


CLASS_NAMES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_image(stem):

    for ext in IMAGE_EXTENSIONS:

        path = (
            SOURCE_IMAGE_DIR
            / f"{stem}{ext}"
        )

        if path.exists():
            return path

    return None


def tile_positions(length):

    """
    Generate full-size tile start positions.

    The final tile is shifted toward the boundary when
    necessary so that no incomplete tile is produced.
    """

    if length <= TILE_SIZE:
        return [0]

    positions = list(
        range(
            0,
            length - TILE_SIZE + 1,
            STRIDE
        )
    )

    final_position = (
        length - TILE_SIZE
    )

    if positions[-1] != final_position:

        positions.append(
            final_position
        )

    return sorted(
        set(positions)
    )


def yolo_to_xyxy(
    x_center,
    y_center,
    width,
    height,
    image_width,
    image_height,
):

    box_width = (
        width * image_width
    )

    box_height = (
        height * image_height
    )

    center_x = (
        x_center * image_width
    )

    center_y = (
        y_center * image_height
    )

    x1 = (
        center_x
        - box_width / 2
    )

    y1 = (
        center_y
        - box_height / 2
    )

    x2 = (
        center_x
        + box_width / 2
    )

    y2 = (
        center_y
        + box_height / 2
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def xyxy_to_yolo(
    x1,
    y1,
    x2,
    y2,
):

    width = (
        x2 - x1
    )

    height = (
        y2 - y1
    )

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    return (
        center_x / TILE_SIZE,
        center_y / TILE_SIZE,
        width / TILE_SIZE,
        height / TILE_SIZE,
    )


# ============================================================
# PRE-FLIGHT CHECK
# ============================================================

print("=" * 80)
print("PCB TRAIN-ONLY 384x384 TILING")
print("=" * 80)

print(
    "Source:",
    SOURCE_DATASET
)

print(
    "Output:",
    OUTPUT_DATASET
)

print(
    "Tile size:",
    TILE_SIZE
)

print(
    "Nominal overlap:",
    f"{OVERLAP * 100:.0f}%"
)

print(
    "Nominal stride:",
    STRIDE
)

print(
    "Minimum retained bbox area:",
    f"{MIN_RETAINED_AREA * 100:.0f}%"
)


if not SOURCE_IMAGE_DIR.exists():

    raise FileNotFoundError(
        SOURCE_IMAGE_DIR
    )


if not SOURCE_LABEL_DIR.exists():

    raise FileNotFoundError(
        SOURCE_LABEL_DIR
    )


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if OUTPUT_DATASET.exists():

    print(
        "\nWARNING:"
        "\nExisting tiled dataset will be replaced:"
    )

    print(
        OUTPUT_DATASET
    )

    shutil.rmtree(
        OUTPUT_DATASET
    )


OUTPUT_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROCESS TRAINING IMAGES ONLY
# ============================================================

manifest_records = []

total_source_images = 0
total_tiles = 0
total_positive_tiles = 0
total_background_tiles = 0

total_original_boxes = 0
total_retained_boxes = 0
total_discarded_fragments = 0

missing_images = []
invalid_labels = []


label_files = sorted(
    SOURCE_LABEL_DIR.glob(
        "*.txt"
    )
)


print(
    "\nTraining label files:",
    len(label_files)
)


for source_index, label_path in enumerate(
    label_files,
    start=1
):

    image_path = find_image(
        label_path.stem
    )

    if image_path is None:

        missing_images.append(
            label_path.name
        )

        continue


    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        missing_images.append(
            image_path.name
        )

        continue


    image_h, image_w = (
        image.shape[:2]
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if (
        image_w < TILE_SIZE
        or image_h < TILE_SIZE
    ):

        raise RuntimeError(
            f"Image smaller than tile size: "
            f"{image_path.name} "
            f"{image_w}x{image_h}"
        )


    total_source_images += 1


    # --------------------------------------------------------
    # Read source YOLO labels
    # --------------------------------------------------------

    source_boxes = []


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
                    label_path.name,
                    line_number,
                    line,
                ]
            )

            continue


        try:

            class_id = int(
                float(parts[0])
            )

            x_center = float(
                parts[1]
            )

            y_center = float(
                parts[2]
            )

            width = float(
                parts[3]
            )

            height = float(
                parts[4]
            )

        except ValueError:

            invalid_labels.append(
                [
                    label_path.name,
                    line_number,
                    line,
                ]
            )

            continue


        if not (
            0 <= class_id
            < len(CLASS_NAMES)
        ):

            invalid_labels.append(
                [
                    label_path.name,
                    line_number,
                    line,
                ]
            )

            continue


        x1, y1, x2, y2 = (
            yolo_to_xyxy(
                x_center,
                y_center,
                width,
                height,
                image_w,
                image_h,
            )
        )


        original_area = (
            max(0, x2 - x1)
            * max(0, y2 - y1)
        )


        if original_area <= 0:

            invalid_labels.append(
                [
                    label_path.name,
                    line_number,
                    line,
                ]
            )

            continue


        source_boxes.append(
            {
                "class_id":
                    class_id,

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2,

                "area":
                    original_area,
            }
        )


    total_original_boxes += (
        len(source_boxes)
    )


    # --------------------------------------------------------
    # Create tile positions
    # --------------------------------------------------------

    x_positions = tile_positions(
        image_w
    )

    y_positions = tile_positions(
        image_h
    )


    for tile_y in y_positions:

        for tile_x in x_positions:

            tile_x2 = (
                tile_x
                + TILE_SIZE
            )

            tile_y2 = (
                tile_y
                + TILE_SIZE
            )


            tile = image[
                tile_y:tile_y2,
                tile_x:tile_x2
            ]


            if tile.shape[0] != TILE_SIZE:

                raise RuntimeError(
                    "Unexpected tile height."
                )


            if tile.shape[1] != TILE_SIZE:

                raise RuntimeError(
                    "Unexpected tile width."
                )


            # ------------------------------------------------
            # Process objects intersecting this tile
            # ------------------------------------------------

            tile_labels = []

            retained_objects = []

            discarded_here = 0


            for object_index, box in enumerate(
                source_boxes
            ):

                intersection_x1 = max(
                    box["x1"],
                    tile_x
                )

                intersection_y1 = max(
                    box["y1"],
                    tile_y
                )

                intersection_x2 = min(
                    box["x2"],
                    tile_x2
                )

                intersection_y2 = min(
                    box["y2"],
                    tile_y2
                )


                intersection_width = max(
                    0,
                    intersection_x2
                    - intersection_x1
                )

                intersection_height = max(
                    0,
                    intersection_y2
                    - intersection_y1
                )


                intersection_area = (
                    intersection_width
                    * intersection_height
                )


                # Object does not overlap tile.
                if intersection_area <= 0:
                    continue


                retained_ratio = (
                    intersection_area
                    / box["area"]
                )


                # --------------------------------------------
                # Discard severe truncation
                # --------------------------------------------

                if retained_ratio < MIN_RETAINED_AREA:

                    discarded_here += 1

                    total_discarded_fragments += 1

                    continue


                # --------------------------------------------
                # Convert clipped box to tile-local coords
                # --------------------------------------------

                local_x1 = (
                    intersection_x1
                    - tile_x
                )

                local_y1 = (
                    intersection_y1
                    - tile_y
                )

                local_x2 = (
                    intersection_x2
                    - tile_x
                )

                local_y2 = (
                    intersection_y2
                    - tile_y
                )


                # Clip numerically
                local_x1 = max(
                    0,
                    min(
                        TILE_SIZE,
                        local_x1
                    )
                )

                local_y1 = max(
                    0,
                    min(
                        TILE_SIZE,
                        local_y1
                    )
                )

                local_x2 = max(
                    0,
                    min(
                        TILE_SIZE,
                        local_x2
                    )
                )

                local_y2 = max(
                    0,
                    min(
                        TILE_SIZE,
                        local_y2
                    )
                )


                if (
                    local_x2 <= local_x1
                    or local_y2 <= local_y1
                ):

                    continue


                xc, yc, bw, bh = (
                    xyxy_to_yolo(
                        local_x1,
                        local_y1,
                        local_x2,
                        local_y2,
                    )
                )


                # --------------------------------------------
                # Final coordinate safety check
                # --------------------------------------------

                values = [
                    xc,
                    yc,
                    bw,
                    bh,
                ]


                if not all(
                    0 <= v <= 1
                    for v in values
                ):

                    raise RuntimeError(
                        f"Invalid normalized box "
                        f"in {image_path.name}"
                    )


                tile_labels.append(
                    f"{box['class_id']} "
                    f"{xc:.8f} "
                    f"{yc:.8f} "
                    f"{bw:.8f} "
                    f"{bh:.8f}"
                )


                retained_objects.append(
                    {
                        "object_index":
                            object_index,

                        "class_id":
                            box["class_id"],

                        "retained_ratio":
                            retained_ratio,
                    }
                )


                total_retained_boxes += 1


            # ------------------------------------------------
            # Tile filename
            # ------------------------------------------------

            tile_stem = (
                f"{image_path.stem}"
                f"__x{tile_x}"
                f"_y{tile_y}"
            )


            tile_image_path = (
                OUTPUT_IMAGE_DIR
                / f"{tile_stem}.png"
            )


            tile_label_path = (
                OUTPUT_LABEL_DIR
                / f"{tile_stem}.txt"
            )


            # ------------------------------------------------
            # Save tile losslessly
            # ------------------------------------------------

            success = cv2.imwrite(
                str(tile_image_path),
                tile
            )


            if not success:

                raise RuntimeError(
                    f"Could not save "
                    f"{tile_image_path}"
                )


            # ------------------------------------------------
            # Save label file
            # Empty file = valid negative/background tile
            # ------------------------------------------------

            with open(
                tile_label_path,
                "w",
                encoding="utf-8"
            ) as f:

                if tile_labels:

                    f.write(
                        "\n".join(
                            tile_labels
                        )
                    )

                    f.write("\n")


            total_tiles += 1


            if tile_labels:

                total_positive_tiles += 1

            else:

                total_background_tiles += 1


            # ------------------------------------------------
            # Manifest record
            # ------------------------------------------------

            manifest_records.append({

                "tile_filename":
                    tile_image_path.name,

                "source_image":
                    image_path.name,

                "source_label":
                    label_path.name,

                "source_width":
                    image_w,

                "source_height":
                    image_h,

                "tile_x":
                    tile_x,

                "tile_y":
                    tile_y,

                "tile_size":
                    TILE_SIZE,

                "nominal_stride":
                    STRIDE,

                "objects_in_tile":
                    len(tile_labels),

                "discarded_fragments":
                    discarded_here,

                "is_background_tile":
                    int(
                        len(tile_labels) == 0
                    ),
            })


    if (
        source_index % 500 == 0
        or source_index
        == len(label_files)
    ):

        print(
            f"Processed "
            f"{source_index}/"
            f"{len(label_files)} "
            f"source images"
        )


# ============================================================
# SAVE MANIFEST
# ============================================================

manifest_df = pd.DataFrame(
    manifest_records
)

manifest_df.to_csv(
    MANIFEST_PATH,
    index=False
)


# ============================================================
# SAVE AUDIT FILES
# ============================================================

if missing_images:

    pd.DataFrame(
        {
            "missing_image":
                missing_images
        }
    ).to_csv(
        OUTPUT_DATASET
        / "missing_images.csv",
        index=False
    )


if invalid_labels:

    pd.DataFrame(
        invalid_labels,
        columns=[
            "label_file",
            "line_number",
            "content",
        ]
    ).to_csv(
        OUTPUT_DATASET
        / "invalid_labels.csv",
        index=False
    )


# ============================================================
# DEVELOPMENT YAML
# ============================================================
#
# Train = tiled train
# Val   = ORIGINAL leakage-free validation
#
# TEST IS INTENTIONALLY NOT INCLUDED.
# ============================================================

yaml_text = f"""# Development-only dataset configuration
# Train: tiled leakage-free training images
# Validation: original leakage-free validation images
# Test intentionally omitted during model development.

train: {str(OUTPUT_IMAGE_DIR).replace(chr(92), "/")}
val: {str(SOURCE_DATASET / "images" / "val").replace(chr(92), "/")}

nc: 6

names:
  0: missing_hole
  1: mouse_bite
  2: open_circuit
  3: short
  4: spur
  5: spurious_copper
"""


with open(
    YAML_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        yaml_text
    )


# ============================================================
# SUMMARY FILE
# ============================================================

summary_path = (
    OUTPUT_DATASET
    / "TILING_SUMMARY.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "PCB TRAIN-ONLY TILING SUMMARY\n"
    )

    f.write(
        "=============================\n\n"
    )

    f.write(
        "SOURCE SPLIT: TRAIN ONLY\n"
    )

    f.write(
        "Validation was NOT tiled.\n"
    )

    f.write(
        "Test was NOT accessed or tiled.\n\n"
    )

    f.write(
        f"Tile size: "
        f"{TILE_SIZE}x{TILE_SIZE}\n"
    )

    f.write(
        f"Nominal overlap: "
        f"{OVERLAP * 100:.1f}%\n"
    )

    f.write(
        f"Nominal stride: "
        f"{STRIDE}\n"
    )

    f.write(
        f"Minimum retained bbox area: "
        f"{MIN_RETAINED_AREA * 100:.1f}%\n\n"
    )

    f.write(
        f"Source images: "
        f"{total_source_images}\n"
    )

    f.write(
        f"Original objects: "
        f"{total_original_boxes}\n"
    )

    f.write(
        f"Generated tiles: "
        f"{total_tiles}\n"
    )

    f.write(
        f"Positive tiles: "
        f"{total_positive_tiles}\n"
    )

    f.write(
        f"Background tiles: "
        f"{total_background_tiles}\n"
    )

    f.write(
        f"Retained tile-object instances: "
        f"{total_retained_boxes}\n"
    )

    f.write(
        f"Discarded truncated fragments: "
        f"{total_discarded_fragments}\n"
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
# FINAL PRINT
# ============================================================

print("\n" + "=" * 80)
print("TILING COMPLETE")
print("=" * 80)

print(
    "Source images:",
    total_source_images
)

print(
    "Original objects:",
    total_original_boxes
)

print(
    "Generated tiles:",
    total_tiles
)

print(
    "Positive tiles:",
    total_positive_tiles
)

print(
    "Background tiles:",
    total_background_tiles
)

print(
    "Retained tile-object instances:",
    total_retained_boxes
)

print(
    "Discarded truncated fragments:",
    total_discarded_fragments
)

print(
    "Missing images:",
    len(missing_images)
)

print(
    "Invalid labels:",
    len(invalid_labels)
)

print("\nDataset:")
print(
    OUTPUT_DATASET
)

print("\nDevelopment YAML:")
print(
    YAML_PATH
)

print("\nManifest:")
print(
    MANIFEST_PATH
)

print(
    "\nIMPORTANT:"
    "\nValidation remains ORIGINAL."
    "\nTest was not touched."
)