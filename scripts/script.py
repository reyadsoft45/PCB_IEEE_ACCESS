from pathlib import Path
import pandas as pd
from collections import defaultdict

DATASET = Path(
    r"C:\PCB_IEEE_ACCESS\dataset\VOC_PCB_CLEAN"
)

manifest = pd.read_csv(
    DATASET / "dataset_manifest.csv"
)

current_files = defaultdict(list)

for split in ["train", "val", "test"]:

    folder = DATASET / "images" / split

    for p in folder.rglob("*"):

        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp"
        }:

            current_files[p.name].append(p)

found_once = 0
missing = []
multiple = []

for filename in manifest["filename"]:

    matches = current_files.get(
        filename, []
    )

    if len(matches) == 1:
        found_once += 1

    elif len(matches) == 0:
        missing.append(filename)

    else:
        multiple.append(
            (filename, matches)
        )

print("=" * 55)
print("MANIFEST vs CURRENT DATASET")
print("=" * 55)

print("Manifest rows:", len(manifest))
print("Found exactly once:", found_once)
print("Missing:", len(missing))
print("Multiple matches:", len(multiple))

print("\nFirst 10 missing:")
for x in missing[:10]:
    print(x)

print("\nFirst 10 multiple matches:")
for filename, paths in multiple[:10]:

    print("\n", filename)

    for p in paths:
        print("   ", p)
