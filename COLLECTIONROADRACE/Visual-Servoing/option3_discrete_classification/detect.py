"""
Color-threshold target detector. Unlike the other two projects, this one
is used ONLY to bootstrap training labels (see generate_pseudo_classes.py)
and for calibrate_color.py's live preview — it is not swapped in and out
at deployment the way detect.py works in the other two options, because
this project's drive.py doesn't use a centroid/size at all at runtime. The
classifier (classifier_model.py) replaces this detector entirely once
trained; get_target_color() only ever runs again if you want to
regenerate training labels.
"""

import cv2
import numpy as np
from config import HSV_LOWER, HSV_UPPER, MIN_TARGET_AREA_FRACTION


def get_target_color(frame, hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER,
                      min_area_fraction=MIN_TARGET_AREA_FRACTION):
    """Classical detector: HSV color thresholding + largest contour.
    Returns cx_norm, area_frac, bbox, mask — or None if nothing found."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    frame_area = h * w
    if area < min_area_fraction * frame_area:
        return None

    x, y, bw, bh = cv2.boundingRect(largest)
    M = cv2.moments(largest)
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return {
        "cx_norm": (cx - w / 2) / (w / 2),
        "cy_norm": (cy - h / 2) / (h / 2),
        "area_frac": area / frame_area,
        "bbox": (x, y, bw, bh),
        "mask": mask,
    }


def draw_debug(frame, target):
    """Draw the bounding box + centroid on a copy of frame, for a HUD/preview."""
    out = frame.copy()
    h, w = out.shape[:2]
    if target is not None:
        x, y, bw, bh = target["bbox"]
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cx_px = int(w / 2 + target["cx_norm"] * (w / 2))
        cy_px = int(h / 2 + target["cy_norm"] * (h / 2))
        cv2.drawMarker(out, (cx_px, cy_px), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(out, f"area={target['area_frac']:.3f} cx={target['cx_norm']:+.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(out, "TARGET NOT FOUND", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.drawMarker(out, (w // 2, h // 2), (255, 255, 0), cv2.MARKER_CROSS, 12, 1)
    return out
