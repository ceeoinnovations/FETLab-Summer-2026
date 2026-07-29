"""
Classical HSV color-threshold target detector.

This is the exact detector that generated the pseudo-labels the offline
RL actor (mode 9) was trained on — so it's also what supplies that
mode's live perception, rather than mixing in a different (neural)
detector at inference time than the one the training data came from.

Same algorithm as the color backend in the option1/option2/option3
projects' detect.py; copied in here rather than imported so this
project has no dependency on any of those folders being present.
"""
import cv2
import numpy as np

from config import HSV_LOWER, HSV_UPPER, MIN_TARGET_AREA_FRACTION


def get_target_color(frame, hsv_lower=HSV_LOWER, hsv_upper=HSV_UPPER,
                      min_area_fraction=MIN_TARGET_AREA_FRACTION):
    """Find the target in a raw BGR frame (as read from cv2.VideoCapture).
    Returns cx_norm, cy_norm, area_frac, bbox, or None if nothing found."""
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
    }
