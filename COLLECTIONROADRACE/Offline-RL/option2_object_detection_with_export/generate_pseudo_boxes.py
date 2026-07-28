"""
Turn existing collect_data.py images into training data for the grid
detector (grid_detector_model.py), using the color detector's bounding
box output as the label — the same pseudo-labeling idea used for the
keypoint-regression project, just keeping the full box instead of
collapsing it down to a centroid + area.

Always uses the color method (get_target_color), never whatever
config.DETECTOR_BACKEND happens to be set to — this script's whole job is
to bootstrap the grid detector's training data, which would be circular
if it tried to use the (not yet trained) grid detector instead.

Spot-check a random sample of the output CSV against the source images
before trusting it, same caveat as generate_pseudo_labels.py: any mistake
the color detector makes gets faithfully copied into the grid detector's
training data.

Usage:
    python generate_pseudo_boxes.py data/images data/pseudo_boxes.csv
"""

import sys
import csv
import cv2
from pathlib import Path
from detect import get_target_color
from grid_detector_model import GRID_SIZE


def main(img_dir, out_csv):
    img_dir = Path(img_dir)
    files = sorted(img_dir.glob("*.jpg"))
    if not files:
        raise SystemExit(f"No .jpg files found in {img_dir}")

    n_visible = 0
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "visible", "grid_row", "grid_col",
                          "x_offset", "y_offset", "w_frac", "h_frac"])
        for i, path in enumerate(files):
            frame = cv2.imread(str(path))
            h, w = frame.shape[:2]
            target = get_target_color(frame)

            if target is None:
                writer.writerow([path.name, 0, -1, -1, 0.0, 0.0, 0.0, 0.0])
                continue

            x, y, bw, bh = target["bbox"]
            cx_frac = (x + bw / 2) / w
            cy_frac = (y + bh / 2) / h
            # Which cell contains the box's CENTER — same "responsibility"
            # rule real YOLO uses. The box itself (w_frac, h_frac below)
            # can extend well beyond this one cell.
            col = min(GRID_SIZE - 1, max(0, int(cx_frac * GRID_SIZE)))
            row = min(GRID_SIZE - 1, max(0, int(cy_frac * GRID_SIZE)))
            x_offset = cx_frac * GRID_SIZE - col   # 0..1 position within that cell
            y_offset = cy_frac * GRID_SIZE - row
            w_frac = bw / w                          # fraction of the WHOLE frame
            h_frac = bh / h

            writer.writerow([path.name, 1, row, col,
                              f"{x_offset:.4f}", f"{y_offset:.4f}",
                              f"{w_frac:.4f}", f"{h_frac:.4f}"])
            n_visible += 1
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(files)}...")

    print(f"\nDone: {n_visible}/{len(files)} frames had a visible target "
          f"({100 * n_visible / len(files):.1f}%).")
    print(f"Wrote {out_csv}")
    if n_visible / len(files) < 0.7:
        print("\nWARNING: the color detector found the target in less than 70% "
              "of frames. Retune HSV_LOWER/HSV_UPPER with calibrate_color.py "
              "before trusting these labels.")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        img_dir, out_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 1:
        img_dir = Path(__file__).parent / "data" / "images"
        out_csv = Path(__file__).parent / "data" / "pseudo_boxes.csv"
        print(f"No arguments given — using defaults:\n  images_dir = {img_dir}\n  out_csv    = {out_csv}\n")
    else:
        raise SystemExit("Usage: python generate_pseudo_boxes.py <images_dir> <out_csv>")
    main(img_dir, out_csv)
