"""Compute image->floor homography from measured floor points, for the
autonomous fetch-and-kick tracking cycle in main.py.

Coordinate system (must match main.py's ANCHORS/HOME_POSITION exactly - NOT
meters, NOT an independent origin): origin at the floor point directly below
ANCHORS[0], +x/+y along the same directions ANCHORS uses. Units: millimeters.
This is what makes image_to_world()'s output directly usable as a move_to()
argument with zero conversion.

Manual workflow:
  1. Mark >= 6 identifiable, non-collinear floor points.
  2. arduino-app-cli app stop ~/ArduinoApps/spider-kicker-test
  3. python3 snapshot.py -> produces a cropped snapshot.jpg.
  4. Read pixel (u, v) for each marked point off snapshot.jpg in any image
     viewer.
  5. Tape-measure each point's real (x, y) in mm from the floor point
     directly below ANCHORS[0], along the same +x/+y directions ANCHORS uses
     (assumes plumb rig posts).
  6. Fill in points.csv, then:
       cd ~/ArduinoApps/spider-kicker-test/python
       python3 calibrate.py
     Check the printed RMSE.
  7. arduino-app-cli app start ~/ArduinoApps/spider-kicker-test - main.py
     loads calibration.npz at boot; the web UI's "Calibrated" field confirms
     it loaded.

Re-run whenever the camera moves, refocuses, is replaced, or the rig frame
is adjusted.

MODEL_INPUT_SIZE below must be kept in sync with main.py's own constant of
the same name - it's the fixed pixel space points.csv was measured against,
so it's recorded here directly rather than queried live from a camera.
"""
import csv
import sys

import cv2
import numpy as np

MODEL_INPUT_SIZE = 480


def load_correspondences(csv_path):
    """Read points.csv -> (image_pts Nx2 float32, world_pts Nx2 float32)."""
    image_pts, world_pts = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            image_pts.append((float(row["image_u"]), float(row["image_v"])))
            world_pts.append((float(row["world_x"]), float(row["world_y"])))

    n = len(image_pts)
    if n < 4:
        raise ValueError(f"Need at least 4 correspondences, got {n}")
    if n < 6:
        print(f"Warning: only {n} points provided; 6 recommended for a solid least-squares fit")

    return (
        np.array(image_pts, dtype=np.float32),
        np.array(world_pts, dtype=np.float32),
    )


def compute_homography(image_pts, world_pts):
    """findHomography image->world. Returns (H, inlier_mask).
    method=0 (all-points least squares) suits clean measured data;
    switch to cv2.RANSAC only if you suspect a bad measurement."""
    H, mask = cv2.findHomography(image_pts, world_pts, method=0)
    if H is None:
        raise RuntimeError("findHomography failed - check for duplicate/collinear points")
    return H, mask


def reprojection_rmse(H, image_pts, world_pts):
    """Map image_pts through H, compare to measured world_pts.
    Returns RMSE in world units (mm) - the calibration quality metric."""
    projected = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - world_pts, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def save_calibration(path, H, image_size, rmse):
    np.savez(
        path,
        H=H,
        H_inv=np.linalg.inv(H),
        image_size=np.array(image_size),
        units="mm",
        rmse=rmse,
    )


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "points.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "calibration.npz"

    image_pts, world_pts = load_correspondences(csv_path)
    H, mask = compute_homography(image_pts, world_pts)
    rmse = reprojection_rmse(H, image_pts, world_pts)

    print(f"Homography computed from {len(image_pts)} points")
    print(f"Reprojection RMSE: {rmse:.1f} mm")
    if rmse > 20.0:
        print("WARNING: RMSE > 20mm - check for a mislabeled point or bad measurement")

    image_size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    save_calibration(out_path, H, image_size, rmse)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
