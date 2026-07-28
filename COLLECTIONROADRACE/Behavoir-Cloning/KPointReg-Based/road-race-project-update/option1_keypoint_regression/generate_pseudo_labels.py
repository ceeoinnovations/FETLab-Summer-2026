"""
Turn any existing collect_data.py dataset into training data for the
perception model (detector_model.py), with zero new manual labeling.

Runs the classical color-threshold detector (detect.py's get_target,
i.e. the same thing calibrate_color.py tunes) over every already-collected
image and records what it found. That becomes the training target for
train_detector.py — the network learns to imitate the color detector,
so it can only ever be as good as the color detector was on this data.
The point of doing this at all is that a trained network can generalize
past the color detector's blind spots (lighting drift, partial occlusion)
*if* the training images actually contain that variety — if every image
here was shot under one fixed lighting setup, don't expect the trained
model to handle a different one better than the threshold did.

Spot-check a random sample of the output CSV against the source images
before trusting it — any systematic mistake the color detector makes
(false positive on background clutter, missing the target when partially
off-screen) gets faithfully learned by the network, not corrected.

Usage:
    python generate_pseudo_labels.py data/images data/pseudo_labels.csv
"""

import sys
import csv
import cv2
from pathlib import Path
from detect import get_target_color


def main(img_dir, out_csv):
    img_dir = Path(img_dir)
    files = sorted(img_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"No .jpg files found in {img_dir}")

    n_visible = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cx_norm", "area_frac", "visible"])
        for i, path in enumerate(files):
            frame = cv2.imread(str(path))
            target = get_target_color(frame)
            if target is not None:
                writer.writerow([path.name, f"{target['cx_norm']:.4f}",
                                  f"{target['area_frac']:.4f}", 1])
                n_visible += 1
            else:
                # cx_norm/area_frac are meaningless when not visible — 0 is
                # just a placeholder; train_detector.py masks these out of
                # the position/size loss and only uses `visible` for them.
                writer.writerow([path.name, 0.0, 0.0, 0])
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)}...")

    print(f"\nDone: {n_visible}/{len(files)} frames had a visible target "
          f"({100 * n_visible / len(files):.1f}%).")
    print(f"Wrote {out_csv}")
    if n_visible / len(files) < 0.7:
        print("\nWARNING: the color detector found the target in less than 70% "
              "of frames. Either a lot of these images genuinely don't contain "
              "the target, or HSV_LOWER/HSV_UPPER in config.py need retuning "
              "with calibrate_color.py before you trust these labels.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        img_dir, out_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        # No arguments — e.g. running via VS Code's Run button rather than a
        # terminal. Default to data/images and data/pseudo_labels.csv next
        # to this script (Path(__file__).parent, not the current working
        # directory, so this works no matter where VS Code's cwd is set).
        img_dir = Path(__file__).parent / "data" / "images"
        out_csv = Path(__file__).parent / "data" / "pseudo_labels.csv"
        print(f"No arguments given — using defaults:\n  images_dir = {img_dir}\n  out_csv    = {out_csv}\n")
    else:
        raise SystemExit("Usage: python generate_pseudo_labels.py <images_dir> <out_csv>")
    main(img_dir, out_csv)
