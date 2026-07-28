"""
Turn existing collect_data.py images into training data for the
classifier (classifier_model.py), by running the color detector over
every image and bucketing its (cx_norm, area_frac, visible) output into
one of config.CATEGORIES using fixed numeric thresholds.

This is the cheapest possible labeling route for this option: no manual
category-by-category sorting of thousands of images, just automatic
bucketing of the same kind of continuous measurement the other two
options use directly.

Usage:
    python generate_pseudo_classes.py data/images data/pseudo_classes.csv
"""

import sys
import csv
import cv2
from pathlib import Path
from detect import get_target_color
from config import CATEGORIES, CX_HARD_TURN_THRESHOLD, CX_SOFT_TURN_THRESHOLD, STOP_AREA_FRACTION


def bucket(target):
    """Map a get_target_color() result (or None) to one of config.CATEGORIES."""
    if target is None:
        return "not_visible"
    if target["area_frac"] >= STOP_AREA_FRACTION:
        return "stop"
    cx = target["cx_norm"]
    if cx <= -CX_HARD_TURN_THRESHOLD:
        return "hard_left"
    if cx <= -CX_SOFT_TURN_THRESHOLD:
        return "soft_left"
    if cx < CX_SOFT_TURN_THRESHOLD:
        return "straight"
    if cx < CX_HARD_TURN_THRESHOLD:
        return "soft_right"
    return "hard_right"


def main(img_dir, out_csv):
    img_dir = Path(img_dir)
    files = sorted(img_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"No .jpg files found in {img_dir}")

    counts = {c: 0 for c in CATEGORIES}
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "category"])
        for i, path in enumerate(files):
            frame = cv2.imread(str(path))
            target = get_target_color(frame)
            category = bucket(target)
            writer.writerow([path.name, category])
            counts[category] += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)}...")

    print(f"\nDone. Wrote {out_csv}\n")
    print("Category distribution:")
    total = len(files)
    for c in CATEGORIES:
        pct = 100 * counts[c] / total if total else 0
        print(f"  {c:12s} {counts[c]:5d}  ({pct:4.1f}%)")

    smallest = min(counts.values())
    if smallest == 0:
        print("\nWARNING: at least one category has ZERO examples. The "
              "classifier cannot learn a category it never sees — collect "
              "more varied data (steeper angles, closer approaches) or "
              "adjust the CX_*_THRESHOLD / STOP_AREA_FRACTION boundaries "
              "in config.py.")
    elif smallest < 0.05 * total:
        print("\nNOTE: some categories are quite rare relative to the "
              "others. train_classifier.py weights samples to compensate, "
              "but more balanced data collection would still help.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        img_dir, out_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        img_dir = Path(__file__).parent / "data" / "images"
        out_csv = Path(__file__).parent / "data" / "pseudo_classes.csv"
        print(f"No arguments given — using defaults:\n  images_dir = {img_dir}\n  out_csv    = {out_csv}\n")
    else:
        raise SystemExit("Usage: python generate_pseudo_classes.py <images_dir> <out_csv>")
    main(img_dir, out_csv)
