"""
Confirm near-duplicate (rotation / lighting variant) leakage across splits
==========================================================================

For every source group that appears in two splits, one image from each split is
compared under all 8 rotations/flips. Images are converted to grayscale, resized
to 128x128 and normalized (zero mean, unit std), so lighting changes do not hide
a match. The score is the best Pearson correlation over the 8 transforms.

A random-pair control (images from different groups) shows what "unrelated"
looks like, so the threshold is justified by data, not chosen by hand.

Usage:
  RunPod : python confirm_variant_leakage.py --root /workspace/PCB_YOLO_PAPER_CLEAN
  Windows: python confirm_variant_leakage.py --root C:\\PCB_IEEE_ACCESS\\dataset\\VOC_PCB_LEAKAGE_FREE

Outputs (in <root>/../leakage_check_<dataset name>/):
  pair_scores.csv, summary.json, top_pairs.png (visual evidence)
"""

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
GROUP_RE = re.compile(
    r"(\d{2})_(missing_hole|mouse_bite|open_circuit|short|spur|spurious_copper)"
    r"_(\d{2})(?:_(\d+))?"
)
THRESHOLD = 0.90


def groups(stem):
    m = GROUP_RE.search(stem)
    if not m:
        return stem, stem
    b, c, i, k = m.groups()
    img = f"{b}_{c}_{i}"
    return (f"{img}_{k}" if k else img), img


def sha256(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_norm(p, size=128):
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if g is None:
        raise RuntimeError(f"cannot read {p}")
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def best_corr(a, b):
    best = -1.0
    for flip in (False, True):
        bb = np.fliplr(b) if flip else b
        for k in range(4):
            r = float((a * np.rot90(bb, k)).mean())
            best = max(best, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root with images/{train,val,test}")
    ap.add_argument("--max-pairs", type=int, default=400, help="pairs sampled per split pair")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    root = Path(args.root)
    out = root.parent / f"leakage_check_{root.name}"
    out.mkdir(exist_ok=True)

    splits = [s for s in ("train", "val", "test") if (root / "images" / s).is_dir()]
    files = {s: sorted(p for p in (root / "images" / s).iterdir()
                       if p.suffix.lower() in IMG_EXTS) for s in splits}

    # ---------------- counts and exact duplicates ----------------
    summary = {"dataset": str(root), "splits": {}, "pairs": {}}
    hashes = {}
    for s in splits:
        h = [sha256(p) for p in files[s]]
        hashes[s] = set(h)
        summary["splits"][s] = {"files": len(h), "unique_hashes": len(set(h)),
                                "within_split_exact_duplicates": len(h) - len(set(h))}
        print(f"{s:5s}: {len(h)} files, {len(set(h))} unique "
              f"({len(h) - len(set(h))} exact duplicates inside the split)")

    by_crop = {s: defaultdict(list) for s in splits}
    by_img = {s: defaultdict(list) for s in splits}
    for s in splits:
        for p in files[s]:
            c, i = groups(p.stem)
            by_crop[s][c].append(p)
            by_img[s][i].append(p)

    # ---------------- control: unrelated pairs ----------------
    all_files = [p for s in splits for p in files[s]]
    control = []
    while len(control) < 300:
        a, b = random.sample(all_files, 2)
        if groups(a.stem)[1] != groups(b.stem)[1]:
            control.append(best_corr(load_norm(a), load_norm(b)))
    control = np.array(control)
    print(f"\nControl (unrelated images): median={np.median(control):.3f} "
          f"p99={np.percentile(control, 99):.3f} max={control.max():.3f}")
    summary["control"] = {"median": float(np.median(control)),
                          "p99": float(np.percentile(control, 99)),
                          "max": float(control.max())}

    # ---------------- shared crop groups across splits ----------------
    rows = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if a not in splits or b not in splits:
            continue
        shared = sorted(set(by_crop[a]) & set(by_crop[b]))
        shared_img = set(by_img[a]) & set(by_img[b])
        exact = len(hashes[a] & hashes[b])
        sample = random.sample(shared, min(len(shared), args.max_pairs))
        scores = []
        for g in sample:
            pa, pb = by_crop[a][g][0], by_crop[b][g][0]
            r = best_corr(load_norm(pa), load_norm(pb))
            scores.append(r)
            rows.append({"split_pair": f"{a}-{b}", "group": g,
                         "file_a": pa.name, "file_b": pb.name, "score": round(r, 4)})
        scores = np.array(scores)
        med = float(np.median(scores)) if len(scores) else None
        confirmed = float(np.mean(scores >= THRESHOLD)) if len(sample) else 0.0
        # files in split b that belong to a crop group also present in split a
        affected_b = sum(len(by_crop[b][g]) for g in shared)
        summary["pairs"][f"{a}-{b}"] = {
            "exact_duplicates": exact,
            "shared_crop_groups": len(shared),
            "shared_original_images": len(shared_img),
            f"{b}_files_in_shared_crop_groups": affected_b,
            f"{b}_files_total": len(files[b]),
            "sampled_pairs": len(sample),
            "median_score": med,
            "fraction_confirmed_near_duplicate": confirmed,
        }
        print(f"\n{a}-{b}: exact={exact} | shared crop groups={len(shared)} | "
              f"{b} files affected={affected_b}/{len(files[b])} "
              f"({100 * affected_b / max(len(files[b]), 1):.1f}%)")
        print(f"   sampled {len(sample)} pairs: median score={med if med is None else round(med, 3)}, "
              f"confirmed near-duplicates (>= {THRESHOLD}) = {100 * confirmed:.1f}%")

    with (out / "pair_scores.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["split_pair", "group", "file_a", "file_b", "score"])
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # ---------------- visual evidence: 6 pairs ----------------
    if rows:
        pick = sorted(rows, key=lambda r: -r["score"])[:6]
        tiles = []
        for r in pick:
            sa, sb = r["split_pair"].split("-")
            ia = cv2.resize(cv2.imread(str(root / "images" / sa / r["file_a"])), (256, 256))
            ib = cv2.resize(cv2.imread(str(root / "images" / sb / r["file_b"])), (256, 256))
            row = np.hstack([ia, np.full((256, 8, 3), 255, np.uint8), ib])
            cv2.putText(row, f"{sa}: {r['file_a'][:34]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(row, f"{sb}: {r['file_b'][:34]}", (268, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(row, f"score {r['score']:.3f}", (268, 248), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(row)
            tiles.append(np.full((8, row.shape[1], 3), 255, np.uint8))
        cv2.imwrite(str(out / "top_pairs.png"), np.vstack(tiles))

    print(f"\nSaved: {out}/summary.json, pair_scores.csv, top_pairs.png")


if __name__ == "__main__":
    main()
