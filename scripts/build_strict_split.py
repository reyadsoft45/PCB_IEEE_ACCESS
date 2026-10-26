"""
Strict leakage-free split builder for the augmented PKU-PCB dataset
===================================================================

Pools ALL images of an existing dataset (train+val+test), removes exact
duplicates, and reassigns splits so that every derived version of one original
PCB image (all crops, rotations, lighting variants) stays in ONE split.

Modes
  group : split by original image (board_class_index), stratified by
          (board, defect class), default 70/15/15.
          -> "unseen images of known board designs"
  board : split by PCB board design (template).
          -> "unseen board designs" (generalization test)

Groups that share an identical file (same SHA-256 under different names) are
merged first, so no exact copy can cross splits.

Usage (Windows):
  python build_strict_split.py --src C:\\PCB_IEEE_ACCESS\\dataset\\VOC_PCB_LEAKAGE_FREE ^
      --out C:\\PCB_IEEE_ACCESS\\dataset\\VOC_PCB_STRICT_GROUP --mode group
  python build_strict_split.py --src C:\\PCB_IEEE_ACCESS\\dataset\\VOC_PCB_LEAKAGE_FREE ^
      --out C:\\PCB_IEEE_ACCESS\\dataset\\VOC_PCB_STRICT_BOARD --mode board

Optional for board mode: --val-boards 05 --test-boards 01,11
"""

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit",
               "short", "spur", "spurious_copper"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
GROUP_RE = re.compile(
    r"(\d{2})_(missing_hole|mouse_bite|open_circuit|short|spur|spurious_copper)"
    r"_(\d{2})(?:_(\d+))?"
)


def sha256(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def collect(src: Path):
    records, unparsed = [], []
    for split in ("train", "val", "test"):
        img_dir = src / "images" / split
        if not img_dir.is_dir():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            lbl = src / "labels" / split / f"{p.stem}.txt"
            if not lbl.exists():
                sys.exit(f"ERROR: missing label for {p}")
            m = GROUP_RE.search(p.stem)
            if not m:
                unparsed.append(p.name)
                continue
            board, cls, idx, crop = m.groups()
            records.append({
                "filename": p.name, "src_image": p, "src_label": lbl,
                "old_split": split, "board": board, "defect_class": cls,
                "image_group": f"{board}_{cls}_{idx}",
                "crop_group": f"{board}_{cls}_{idx}_{crop}" if crop else f"{board}_{cls}_{idx}",
            })
    if unparsed:
        sys.exit(f"ERROR: {len(unparsed)} filenames do not match the PKU naming pattern, "
                 f"e.g. {unparsed[:5]}")
    return records


def dedupe_and_merge(records):
    print("Hashing images ...", flush=True)
    for r in records:
        r["sha256"] = sha256(r["src_image"])
        r["label_sha256"] = hashlib.sha256(r["src_label"].read_bytes()).hexdigest()

    uf = UnionFind()
    by_hash = defaultdict(list)
    for r in records:
        uf.find(r["image_group"])
        by_hash[r["sha256"]].append(r)

    kept, dropped, label_conflicts = [], 0, []
    for h, rs in by_hash.items():
        for r in rs[1:]:
            uf.union(rs[0]["image_group"], r["image_group"])
        if len({r["label_sha256"] for r in rs}) > 1:
            label_conflicts.append([r["filename"] for r in rs])
        rs_sorted = sorted(rs, key=lambda r: r["filename"])
        kept.append(rs_sorted[0])
        dropped += len(rs) - 1

    for r in kept:
        r["split_group"] = uf.find(r["image_group"])
    all_groups = {r["image_group"] for r in records}
    merged = len(all_groups) - len({uf.find(g) for g in all_groups})
    print(f"Files: {len(records)} | exact duplicates removed: {dropped} | "
          f"kept: {len(kept)} | groups merged by shared hashes: {merged}")
    if label_conflicts:
        print(f"WARNING: {len(label_conflicts)} duplicate sets have DIFFERENT labels; "
              f"the alphabetically first file was kept. Examples: {label_conflicts[:3]}")
    return kept, dropped, label_conflicts


def assign_group_mode(records, ratios, seed):
    rng = random.Random(seed)
    group_cell = {}
    for r in records:
        group_cell.setdefault(r["split_group"], (r["board"], r["defect_class"]))
    cells = defaultdict(list)
    for g, cell in group_cell.items():
        cells[cell].append(g)

    split_of = {}
    for cell in sorted(cells):
        gs = sorted(cells[cell])
        rng.shuffle(gs)
        n = len(gs)
        n_test = round(n * ratios[2])
        n_val = round(n * ratios[1])
        for i, g in enumerate(gs):
            split_of[g] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return split_of, {}


def assign_board_mode(records, ratios, seed, val_boards, test_boards):
    per_board = Counter(r["board"] for r in records)
    boards = sorted(per_board)
    print("Images per board:", dict(per_board))
    if not (val_boards and test_boards):
        rng = random.Random(seed)
        order = boards[:]
        rng.shuffle(order)
        total = sum(per_board.values())
        test_boards, val_boards, acc = [], [], 0
        for b in order:
            if acc < ratios[2] * total:
                test_boards.append(b)
                acc += per_board[b]
        acc = 0
        for b in order:
            if b in test_boards:
                continue
            if acc < ratios[1] * total:
                val_boards.append(b)
                acc += per_board[b]
    overlap = set(val_boards) & set(test_boards)
    if overlap:
        sys.exit(f"ERROR: boards in both val and test: {overlap}")
    split_of = {}
    for r in records:
        b = r["board"]
        split_of[r["split_group"]] = ("test" if b in test_boards
                                      else "val" if b in val_boards else "train")
    # a merged group spanning several boards must not cross splits
    return split_of, {"val_boards": sorted(val_boards), "test_boards": sorted(test_boards)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["group", "board"], default="group")
    ap.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-boards", default="")
    ap.add_argument("--test-boards", default="")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    ratios = [float(x) for x in args.ratios.split(",")]
    if out.exists():
        sys.exit(f"ERROR: {out} already exists. Delete or rename it first.")

    records = collect(src)
    records, dropped, conflicts = dedupe_and_merge(records)

    if args.mode == "group":
        split_of, info = assign_group_mode(records, ratios, args.seed)
    else:
        vb = [b for b in args.val_boards.split(",") if b]
        tb = [b for b in args.test_boards.split(",") if b]
        split_of, info = assign_board_mode(records, ratios, args.seed, vb, tb)
    for r in records:
        r["split"] = split_of[r["split_group"]]

    # ---------------- verification before writing ----------------
    def overlap(key):
        sets = defaultdict(set)
        for r in records:
            sets[r["split"]].add(r[key])
        return {f"{a}-{b}": len(sets[a] & sets[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}

    checks = {"sha256": overlap("sha256"), "crop_group": overlap("crop_group"),
              "image_group": overlap("image_group"), "split_group": overlap("split_group")}
    if args.mode == "board":
        checks["board"] = overlap("board")
    bad = {k: v for k, v in checks.items() if any(v.values())}
    if bad:
        sys.exit(f"ERROR: overlap after assignment: {bad}")

    # ---------------- write dataset ----------------
    for s in ("train", "val", "test"):
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    for r in records:
        shutil.copy2(r["src_image"], out / "images" / r["split"] / r["filename"])
        shutil.copy2(r["src_label"], out / "labels" / r["split"] / f"{Path(r['filename']).stem}.txt")

    names = {i: n for i, n in enumerate(CLASS_NAMES)}
    full = {"path": str(out), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": names}
    dev = {k: v for k, v in full.items() if k != "test"}
    (out / "data.yaml").write_text(yaml.safe_dump(full, sort_keys=False), encoding="utf-8")
    (out / "data_dev.yaml").write_text(
        "# development YAML: no test split\n" + yaml.safe_dump(dev, sort_keys=False), encoding="utf-8")

    fields = ["filename", "split", "old_split", "board", "defect_class",
              "image_group", "crop_group", "split_group", "sha256"]
    with (out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(records, key=lambda r: (r["split"], r["filename"])))

    # ---------------- report ----------------
    split_counts = Counter(r["split"] for r in records)
    obj_counts = {s: Counter() for s in ("train", "val", "test")}
    for r in records:
        for line in r["src_label"].read_text(encoding="utf-8").split("\n"):
            if line.strip():
                obj_counts[r["split"]][CLASS_NAMES[int(line.split()[0])]] += 1
    report = {
        "mode": args.mode, "seed": args.seed, "ratios": ratios, **info,
        "exact_duplicates_removed": dropped, "label_conflicts": len(conflicts),
        "images": dict(split_counts),
        "groups": {s: len({r["split_group"] for r in records if r["split"] == s})
                   for s in ("train", "val", "test")},
        "objects_per_class": {s: dict(obj_counts[s]) for s in obj_counts},
        "overlap_checks": checks,
        "status": "PASS",
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nMode: {args.mode}" + (f" | val boards {info['val_boards']} | test boards "
                                     f"{info['test_boards']}" if info else ""))
    total = sum(split_counts.values())
    for s in ("train", "val", "test"):
        print(f"  {s:5s}: {split_counts[s]:6d} images ({100 * split_counts[s] / total:4.1f}%) | "
              f"{report['groups'][s]:5d} groups | objects {sum(obj_counts[s].values())}")
    print("\nObjects per class (train / val / test):")
    for c in CLASS_NAMES:
        print(f"  {c:16s} {obj_counts['train'][c]:6d} / {obj_counts['val'][c]:5d} / {obj_counts['test'][c]:5d}")
    print("\nOverlap checks (all must be 0):")
    for k, v in checks.items():
        print(f"  {k:12s} {v}")
    print(f"\nStatus: PASS | written to {out}")


if __name__ == "__main__":
    main()
