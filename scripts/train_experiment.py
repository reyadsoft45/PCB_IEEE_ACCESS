"""
Unified training script for the strict leakage-free splits
==========================================================

One script for every experiment, on Windows (RTX 3060) or RunPod (RTX 3090).

  --variant baseline : train on original 600x600 images
  --variant tiled    : train on 384x384 train-only tiles (created once, cached),
                       validate on ORIGINAL val images

Compute matching for tiled runs is automatic: each 600x600 image gives 4 tiles,
so a tiled run with E epochs = baseline with 4E epochs in optimizer steps.
Warmup and no-mosaic phases are scaled to the same number of steps.

Examples
  python train_experiment.py --data-root /workspace/VOC_PCB_STRICT_GROUP --model yolo11n.pt --variant baseline --epochs 100
  python train_experiment.py --data-root /workspace/VOC_PCB_STRICT_GROUP --model yolo11s.pt --variant tiled --epochs 25

Run name is generated automatically, e.g. STRICT_GROUP__yolo11s__tiled384__e25__s42
The TEST split is never used here.
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml

CLASS_NAMES = ["missing_hole", "mouse_bite", "open_circuit",
               "short", "spur", "spurious_copper"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
GROUP_RE = re.compile(
    r"(\d{2})_(missing_hole|mouse_bite|open_circuit|short|spur|spurious_copper)_(\d{2})"
)

TILE_SIZE, OVERLAP, MIN_RETAINED_AREA = 384, 0.25, 0.50
STRIDE = int(TILE_SIZE * (1 - OVERLAP))          # 288 -> 4 tiles per 600x600 image
BASELINE_WARMUP_EPOCHS, BASELINE_CLOSE_MOSAIC_FRACTION = 3.0, 0.10


# ---------------------------------------------------------------------
def list_images(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78, flush=True)


def leakage_precheck(root: Path):
    """Fast name-based check: no original-image group may appear in two splits."""
    groups = defaultdict(set)
    for s in ("train", "val", "test"):
        d = root / "images" / s
        if d.is_dir():
            for p in list_images(d):
                m = GROUP_RE.search(p.stem)
                groups[s].add(m.group(0) if m else p.stem)
    bad = {f"{a}-{b}": len(groups[a] & groups[b])
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
           if groups[a] & groups[b]}
    if bad:
        sys.exit(f"ERROR: original-image groups shared across splits: {bad}. "
                 f"This is not a strict split. Do not train.")
    print("Leakage pre-check: PASS (no shared original-image groups)")


def tile_positions(n):
    if n <= TILE_SIZE:
        return [0]
    pos = list(range(0, n - TILE_SIZE + 1, STRIDE))
    if pos[-1] != n - TILE_SIZE:
        pos.append(n - TILE_SIZE)
    return sorted(set(pos))


def make_tiles(src_root: Path, tiled_root: Path):
    done = tiled_root / "TILING_DONE.json"
    if done.exists():
        print(f"Tiles already exist: {tiled_root}")
        return json.loads(done.read_text())
    if tiled_root.exists():
        sys.exit(f"ERROR: {tiled_root} exists but is incomplete. Delete it and rerun.")
    oi, ol = tiled_root / "images" / "train", tiled_root / "labels" / "train"
    oi.mkdir(parents=True)
    ol.mkdir(parents=True)

    st, t0 = Counter(), time.time()
    imgs = list_images(src_root / "images" / "train")
    for k, p in enumerate(imgs, 1):
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            sys.exit(f"ERROR: cannot read {p}")
        H, W = im.shape[:2]
        if H < TILE_SIZE or W < TILE_SIZE:
            sys.exit(f"ERROR: image smaller than tile: {p.name}")
        boxes = []
        txt = (src_root / "labels" / "train" / f"{p.stem}.txt").read_text(encoding="utf-8")
        for line in txt.split("\n"):
            if not line.strip():
                continue
            c, xc, yc, w, h = line.split()
            xc, yc, w, h = float(xc) * W, float(yc) * H, float(w) * W, float(h) * H
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            if w * h > 0:
                boxes.append((int(c), x1, y1, x2, y2, w * h))
        st["source_images"] += 1
        st["source_objects"] += len(boxes)
        for ty in tile_positions(H):
            for tx in tile_positions(W):
                lines = []
                for c, x1, y1, x2, y2, a in boxes:
                    ix1, iy1 = max(x1, tx), max(y1, ty)
                    ix2, iy2 = min(x2, tx + TILE_SIZE), min(y2, ty + TILE_SIZE)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter <= 0:
                        continue
                    if inter / a < MIN_RETAINED_AREA:
                        st["discarded_fragments"] += 1
                        continue
                    lx1, ly1 = ix1 - tx, iy1 - ty
                    lx2, ly2 = ix2 - tx, iy2 - ty
                    lines.append(f"{c} {(lx1 + lx2) / 2 / TILE_SIZE:.6f} {(ly1 + ly2) / 2 / TILE_SIZE:.6f} "
                                 f"{(lx2 - lx1) / TILE_SIZE:.6f} {(ly2 - ly1) / TILE_SIZE:.6f}")
                stem = f"{p.stem}__x{tx}_y{ty}"
                if not cv2.imwrite(str(oi / f"{stem}.png"), im[ty:ty + TILE_SIZE, tx:tx + TILE_SIZE]):
                    sys.exit(f"ERROR: cannot write {stem}.png")
                (ol / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                st["tiles"] += 1
                st["objects"] += len(lines)
                st["background_tiles"] += int(not lines)
        if k % 1000 == 0 or k == len(imgs):
            print(f"  tiled {k}/{len(imgs)} images -> {st['tiles']} tiles ({time.time() - t0:.0f}s)", flush=True)

    st["tiles_per_image"] = st["tiles"] / max(st["source_images"], 1)
    done.write_text(json.dumps(st, indent=2))
    print("Tiling:", dict(st))
    return st


# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="strict split root (images/, labels/)")
    ap.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt, yolo11s.pt or a model .yaml")
    ap.add_argument("--variant", choices=["baseline", "tiled"], default="baseline")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--results", default=None, help="results folder (default: <data-root>/../results)")
    args = ap.parse_args()

    root = Path(args.data_root).resolve()
    results = Path(args.results).resolve() if args.results else root.parent / "results"
    tag = root.name.replace("VOC_PCB_", "")
    model_tag = Path(args.model).stem
    variant_tag = "tiled384" if args.variant == "tiled" else "full"
    run_name = f"{tag}__{model_tag}__{variant_tag}__e{args.epochs}__s{args.seed}"
    run_dir = results / run_name
    if run_dir.exists():
        sys.exit(f"ERROR: {run_dir} exists. Do not overwrite a previous experiment.")

    banner(f"EXPERIMENT {run_name}")
    for s in ("train", "val"):
        if not (root / "images" / s).is_dir():
            sys.exit(f"ERROR: missing {root / 'images' / s}")
    leakage_precheck(root)

    n_train = len(list_images(root / "images" / "train"))
    if args.variant == "tiled":
        tiled_root = root.parent / f"{root.name}_TILED384"
        st = make_tiles(root, tiled_root)
        train_path = tiled_root / "images" / "train"
        tiles_per_img = st["tiles_per_image"]
        warmup = BASELINE_WARMUP_EPOCHS / tiles_per_img
        n_train_items = st["tiles"]
    else:
        train_path = root / "images" / "train"
        warmup = BASELINE_WARMUP_EPOCHS
        n_train_items = n_train
    close_mosaic = max(1, math.ceil(args.epochs * BASELINE_CLOSE_MOSAIC_FRACTION))
    steps = math.ceil(n_train_items / args.batch) * args.epochs

    data_yaml = results / f"{run_name}__data.yaml"
    results.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(yaml.safe_dump({
        "train": str(train_path), "val": str(root / "images" / "val"),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)}}, sort_keys=False), encoding="utf-8")

    print(f"train items: {n_train_items} | epochs {args.epochs} | ~{steps} optimizer steps | "
          f"warmup_epochs {warmup:.2f} | close_mosaic {close_mosaic}")

    import torch
    import ultralytics
    from ultralytics import YOLO
    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_args = dict(
        data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=0, workers=args.workers, cache=False, amp=True, optimizer="auto",
        warmup_epochs=warmup, close_mosaic=close_mosaic, patience=max(20, args.epochs // 5),
        seed=args.seed, deterministic=True, val=True, plots=True, save=True,
        save_period=max(1, args.epochs // 10), project=str(results), name=run_name, exist_ok=False,
    )
    t0 = time.time()
    YOLO(args.model).train(**train_args)
    hours = (time.time() - t0) / 3600

    banner("FINAL VALIDATION: best.pt on ORIGINAL val")
    best = YOLO(str(run_dir / "weights" / "best.pt"))
    m = best.val(data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
                 plots=True, project=str(results), name=run_name + "__FINAL_VAL")
    summary = {
        "run": run_name, "dataset": str(root), "model": args.model, "variant": args.variant,
        "train_items": n_train_items, "optimizer_steps": steps, "train_hours": round(hours, 2),
        "precision": float(m.box.mp), "recall": float(m.box.mr),
        "mAP50": float(m.box.map50), "mAP50_95": float(m.box.map),
        "per_class_AP50_95": {best.names[i]: float(m.box.maps[i]) for i in best.names},
        "ultralytics": ultralytics.__version__,
        "train_args": {k: v for k, v in train_args.items() if k not in ("data", "project")},
    }
    (run_dir / "FINAL_VAL_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmAP50-95 {summary['mAP50_95']:.4f} | mAP50 {summary['mAP50']:.4f} | "
          f"P {summary['precision']:.4f} | R {summary['recall']:.4f} | {hours:.1f} h")
    for k, v in summary["per_class_AP50_95"].items():
        print(f"  {k:16s} {v:.4f}")


if __name__ == "__main__":
    main()
