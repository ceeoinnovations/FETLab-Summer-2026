"""Pure image-space helpers shared between main.py (the running app) and the
offline calibration tooling (snapshot.py). No camera/motor/network I/O of its
own, so it's safe to import from a standalone script without triggering
main.py's module-level LEGO motor/camera connections.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def pad_to_square(frame, size):
    """Pad `frame` with blank borders to a centered `size` x `size` square.
    Keeps every original pixel (unlike cropping); the frame is just centered
    on a larger blank canvas. Falls back to the frame unpadded (with a
    one-time warning) if the capture ever comes back larger than `size` in
    either dimension. Currently unused by the runtime (center_crop() below
    measured better) - kept for easy toggling back, see main.py's
    MODEL_INPUT_SIZE comment for why."""
    h, w = frame.shape[:2]
    if h > size or w > size:
        logger.warning(f"Frame {w}x{h} larger than size={size} - using it unpadded.")
        return frame
    canvas = np.zeros((size, size, frame.shape[2]), dtype=frame.dtype)
    top = (size - h) // 2
    left = (size - w) // 2
    canvas[top:top + h, left:left + w] = frame
    return canvas


def center_crop(frame, size):
    """Crop a centered `size` x `size` square out of `frame`. Falls back to
    the frame uncropped (with a one-time warning) if the capture ever comes
    back smaller than `size` in either dimension."""
    h, w = frame.shape[:2]
    if h < size or w < size:
        logger.warning(f"Frame {w}x{h} smaller than size={size} - using it uncropped.")
        return frame
    top = (h - size) // 2
    left = (w - size) // 2
    return frame[top:top + size, left:left + size].copy()


def bottom_center(bbox_xyxy):
    """Floor-contact point of an (x1, y1, x2, y2) bbox: bottom-center, in
    pixels - a better proxy for an object's actual floor position than its
    bbox centroid, especially for objects with real height. Matches
    personplacement-test's bottom_center() convention (there expressed in
    xywh form; this project's detections report xyxy via
    'bounding_box_xyxy')."""
    x1, y1, x2, y2 = bbox_xyxy
    return (x1 + x2) / 2.0, y2
