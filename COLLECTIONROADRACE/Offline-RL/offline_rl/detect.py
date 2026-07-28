"""
Classical HSV color-threshold target detector.

Unlike the other option* projects, there's no learned-detector alternative
here and no config.DETECTOR_BACKEND switch — this project always uses the
color detector, both to generate training pseudo-labels
(generate_pseudo_labels.py) and for live perception (drive.py). That's a
deliberate choice, not a missing feature: the actor learns a mapping FROM
whatever detector produced its training data, so using a different
detector at inference time — even a "better" one — would feed it inputs
shaped differently than what it learned to interpret.

See calibrate_color.py to tune HSV_LOWER/HSV_UPPER for your camera/lighting.
"""

import cv2
import numpy as np

from config import HSV_LOWER, HSV_UPPER, MIN_TARGET_AREA_FRACTION


def get_target_color(frame, hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER,
                      min_area_fraction=MIN_TARGET_AREA_FRACTION):
    """Find the target in a raw BGR frame (as read from cv2.VideoCapture).
    Returns a dict with cx_norm, cy_norm, area_frac, bbox, mask — or None
    if nothing was found."""
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
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    return {
        "cx_norm": (cx - w / 2) / (w / 2),
        "cy_norm": (cy - h / 2) / (h / 2),
        "area_frac": area / frame_area,
        "bbox": (x, y, bw, bh),
        "mask": mask,
    }


# Alias matching the other option* projects' detect.py public API, so
# drive.py can call get_target() regardless of which project it's in.
get_target = get_target_color


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
