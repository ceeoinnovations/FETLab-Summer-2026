"""
Turn collect_data.py's images into the compact-state pseudo-labels
train_offline_rl.py trains on: (cx_norm, cy_norm, area_frac, visible),
via the color detector — same idea as the other option* projects'
generate_pseudo_boxes.py / generate_pseudo_labels.py, just producing this
project's 4-dim state instead of a full bounding box or keypoint.

Always uses the color detector directly (this project has no other
detector backend — see detect.py's docstring for why).

Spot-check a random sample of the output CSV against the source images
before trusting it — any mistake the color detector makes gets faithfully
copied into the actor's training data.

Usage:
    python generate_pseudo_labels.py data/images data/pseudo_labels.csv
"""

import sys
import csv
import cv2
from pathlib import Path
from detect import get_target_color


def main(img_dir, labels_csv, out_csv):
    img_dir = Path(img_dir)
    labels_csv = Path(labels_csv)
    if not labels_csv.exists():
        raise SystemExit(f"Could not find {labels_csv} — run collect_data.py first.")

    with open(labels_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{labels_csv} has no rows.")
    if "session_id" not in rows[0]:
        raise SystemExit(f"{labels_csv} has no session_id column — this project needs "
                          "collect_data.py's session markers to build valid transitions "
                          "(so a transition never crosses from the end of one drive into "
                          "the start of the next).")

    n_visible = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "cx_norm", "cy_norm", "area_frac", "visible", "session_id"])
        for i, row in enumerate(rows):
            frame = cv2.imread(str(img_dir / row["filename"]))
            if frame is None:
                writer.writerow([row["filename"], 0.0, 0.0, 0.0, 0, row["session_id"]])
                continue
            target = get_target_color(frame)
            if target is None:
                writer.writerow([row["filename"], 0.0, 0.0, 0.0, 0, row["session_id"]])
            else:
                n_visible += 1
                writer.writerow([row["filename"], round(target["cx_norm"], 4),
                                  round(target["cy_norm"], 4), round(target["area_frac"], 4),
                                  1, row["session_id"]])
            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1}/{len(rows)}")

    print(f"\nDone: {n_visible}/{len(rows)} frames had a visible target "
          f"({100 * n_visible / len(rows):.1f}%).")
    print(f"Wrote {out_csv}")
    if n_visible / len(rows) < 0.7:
        print("\nWARNING: the color detector found the target in less than 70% "
              "of frames. Retune HSV_LOWER/HSV_UPPER with calibrate_color.py "
              "before trusting these labels.")
    print("\nNext step: python check_data_diversity.py")


if __name__ == "__main__":
    default_images = Path(__file__).parent / "data" / "images"
    default_out = Path(__file__).parent / "data" / "pseudo_labels.csv"

    if len(sys.argv) == 3:
        img_dir, out_csv = Path(sys.argv[1]), sys.argv[2]
    elif len(sys.argv) == 1:
        img_dir, out_csv = default_images, default_out
        print(f"No arguments given — using defaults:\n  images_dir = {img_dir}\n"
              f"  out_csv    = {out_csv}\n")
    else:
        raise SystemExit("Usage: python generate_pseudo_labels.py <images_dir> <out_csv>")

    labels_csv = img_dir.parent / "labels.csv"
    main(img_dir, labels_csv, out_csv)
