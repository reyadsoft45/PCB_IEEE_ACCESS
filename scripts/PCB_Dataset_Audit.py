
from pathlib import Path
from collections import defaultdict
import hashlib
import json

ROOT = Path(
    r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_CLEAN"
)

OUT = Path(
    r"C:\PCB_IEEE_ACCESS\audit_results"
)
OUT.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]
EXT = {".jpg", ".jpeg", ".png", ".bmp"}

report = {}
hashes = defaultdict(list)

for split in SPLITS:
    image_dir = ROOT / "images" / split
    label_dir = ROOT / "labels" / split

    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    if not label_dir.is_dir():
        raise FileNotFoundError(label_dir)

    images = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in EXT
    )

    labels = list(label_dir.rglob("*.txt"))
    missing = []
    invalid = []

    for image in images:
        relative = image.relative_to(image_dir)
        label = label_dir / relative.with_suffix(".txt")

        if not label.exists():
            missing.append(str(relative))
        else:
            for line_no, line in enumerate(
                label.read_text(
                    encoding="utf-8"
                ).splitlines(), 1
            ):
                parts = line.split()

                try:
                    values = list(map(float, parts))

                    valid = (
                        len(values) == 5
                        and values[0].is_integer()
                        and 0 <= values[0] < 6
                        and 0 <= values[1] <= 1
                        and 0 <= values[2] <= 1
                        and 0 < values[3] <= 1
                        and 0 < values[4] <= 1
                        and values[1] - values[3]/2 >= -1e-6
                        and values[1] + values[3]/2 <= 1+1e-6
                        and values[2] - values[4]/2 >= -1e-6
                        and values[2] + values[4]/2 <= 1+1e-6
                    )

                    if not valid:
                        invalid.append(
                            f"{label}:{line_no}"
                        )

                except ValueError:
                    invalid.append(
                        f"{label}:{line_no}"
                    )

        h = hashlib.sha256()

        with image.open("rb") as f:
            for chunk in iter(
                lambda: f.read(1024*1024),
                b""
            ):
                h.update(chunk)

        hashes[h.hexdigest()].append(
            (split, str(image))
        )

    report[split] = {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": missing,
        "invalid_annotations": invalid
    }

    print(f"\n{split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Missing labels:", len(missing))
    print("Invalid annotations:", len(invalid))

# Exact cross-split duplicate check
duplicates = {
    h: locations
    for h, locations in hashes.items()
    if len(set(s for s, _ in locations)) > 1
}

report["cross_split_exact_duplicates"] = duplicates

print("\nCross-split exact duplicates:",
      len(duplicates))

with open(
    OUT / "dataset_audit.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(report, f, indent=2)

print("\nReport:", OUT / "dataset_audit.json")
