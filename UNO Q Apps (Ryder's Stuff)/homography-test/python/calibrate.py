"""Compute image->floor homography from 6 measured ground points.

Coordinate system (pin this down before measuring anything):
  Origin: robot base center, projected straight down to the floor.
  X: forward (direction robot faces), Y: left of the robot. Units: meters.

Run this manually on the board after filling in points.csv -- it is a
one-off calibration step, not the App Lab entry point:

    cd ~/ArduinoApps/homography-test/python
    python3 calibrate.py

Re-run whenever the USB webcam is moved, refocused, or replaced.
"""
import csv
import sys

import cv2
import numpy as np

CAM_INDEX = 0  # /dev/video0 -- USB webcam


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
    Returns RMSE in world units (meters) -- the calibration quality metric."""
    projected = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(projected - world_pts, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def save_calibration(path, H, image_size, rmse):
    np.savez(
        path,
        H=H,
        H_inv=np.linalg.inv(H),
        image_size=np.array(image_size),
        units="meters",
        rmse=rmse,
    )


def get_camera_frame_size(cam_index=CAM_INDEX):
    """Open the USB webcam briefly to read its native capture resolution,
    so calibration.npz records the size locate.py should expect."""
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {cam_index}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Camera opened but frame grab failed")
    h, w = frame.shape[:2]
    return (w, h)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "points.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "calibration.npz"

    image_pts, world_pts = load_correspondences(csv_path)
    H, mask = compute_homography(image_pts, world_pts)
    rmse = reprojection_rmse(H, image_pts, world_pts)

    print(f"Homography computed from {len(image_pts)} points")
    print(f"Reprojection RMSE: {rmse:.4f} m")
    if rmse > 0.05:
        print("WARNING: RMSE > 5cm - check for a mislabeled point or bad measurement")

    image_size = get_camera_frame_size()
    print(f"Camera frame size: {image_size}")

    save_calibration(out_path, H, image_size, rmse)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
