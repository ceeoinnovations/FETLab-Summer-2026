"""Map an image pixel to a floor-plane position using a pre-computed homography.

calibration.npz is produced by calibrate.py (run once on the board after
filling in points.csv with measured floor points) and re-used here at
runtime. Re-run calibrate.py whenever the camera is moved, refocused, or
replaced.
"""
import cv2
import numpy as np


def load_calibration(path):
    """Return dict with H, H_inv, image_size, units, rmse."""
    data = np.load(path)
    return {
        "H": data["H"],
        "H_inv": data["H_inv"],
        "image_size": tuple(int(v) for v in data["image_size"]),
        "units": str(data["units"]),
        "rmse": float(data["rmse"]),
    }


def image_to_world(uv, H):
    """Map one pixel (u, v) to floor coords (x, y) via the homography H."""
    pt = np.array([[uv]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(world[0]), float(world[1])
