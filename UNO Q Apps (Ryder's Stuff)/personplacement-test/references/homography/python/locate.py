"""Detect an object and report its floor position using calibration.npz.

Imported by main.py as the App Lab entry point's detection logic. Can also
be run standalone on the board for testing, before wiring it into the app:

    python3 locate.py [aruco|color]
"""
import sys
import time

import cv2
import numpy as np

CAM_INDEX = 0  # /dev/video0 -- USB webcam; bump if you have more than one camera


def load_calibration(path="calibration.npz"):
    """Return dict with H, H_inv, image_size, units, rmse."""
    data = np.load(path)
    return {
        "H": data["H"],
        "H_inv": data["H_inv"],
        "image_size": tuple(int(v) for v in data["image_size"]),
        "units": str(data["units"]),
        "rmse": float(data["rmse"]),
    }


_cap = None


def get_frame():
    """Grab one BGR frame from the USB webcam. Opens the device on first
    call and keeps it open for the life of the process."""
    global _cap
    if _cap is None:
        _cap = cv2.VideoCapture(CAM_INDEX)
        if not _cap.isOpened():
            raise RuntimeError(f"Could not open camera index {CAM_INDEX}")
    ok, frame = _cap.read()
    if not ok:
        raise RuntimeError("Frame grab failed")
    return frame


def release_frame_source():
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None


def check_frame_size(frame, calib):
    """The board's USB webcam must still be at the resolution it was
    calibrated at -- a changed resolution invalidates H."""
    h, w = frame.shape[:2]
    if (w, h) != calib["image_size"]:
        raise RuntimeError(
            f"Frame size {(w, h)} != calibration size {calib['image_size']}; "
            "re-run calibrate.py with the current camera settings"
        )


# --- swappable detectors: each returns the floor-contact pixel (u,v) or None ---

def detect_aruco(frame, dictionary=cv2.aruco.DICT_4X4_50):
    """Detect first ArUco marker; return its center pixel (mean of 4 corners)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    d = cv2.aruco.getPredefinedDictionary(dictionary)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None
    c = corners[0][0]  # first marker, 4 corners x (u,v)
    u, v = c.mean(axis=0)
    return float(u), float(v)


def detect_color_blob(frame, hsv_low, hsv_high, flat_object=False):
    """inRange in HSV -> morphological open/close -> largest contour.
    Returns the CONTACT point: bottom-center of the bbox for a 3D object
    standing on the floor, or the centroid if the object itself is flat."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None

    if flat_object:
        m = cv2.moments(largest)
        if m["m00"] == 0:
            return None
        return m["m10"] / m["m00"], m["m01"] / m["m00"]

    x, y, w, h = cv2.boundingRect(largest)
    return float(x + w / 2), float(y + h)  # bottom-center = floor contact


def image_to_world(uv, H):
    """Map one pixel to floor coords via cv2.perspectiveTransform."""
    pt = np.array([[uv]], dtype=np.float32)
    world = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(world[0]), float(world[1])


def draw_debug_overlay(frame, calib, uv=None, world_xy=None, grid_step=0.25, grid_extent=2.0):
    """Draw the reprojected world grid (via H_inv) plus the detected point,
    so you can eyeball whether the floor mapping looks sane."""
    out = frame.copy()
    H_inv = calib["H_inv"]
    steps = np.arange(-grid_extent, grid_extent + 1e-6, grid_step)

    for x in steps:
        pts = np.array([[[x, y]] for y in steps], dtype=np.float32)
        img_pts = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
        for p1, p2 in zip(img_pts[:-1], img_pts[1:]):
            cv2.line(out, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 1)
    for y in steps:
        pts = np.array([[[x, y]] for x in steps], dtype=np.float32)
        img_pts = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
        for p1, p2 in zip(img_pts[:-1], img_pts[1:]):
            cv2.line(out, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 255, 0), 1)

    if uv is not None:
        cv2.circle(out, (int(uv[0]), int(uv[1])), 6, (0, 0, 255), -1)
    if world_xy is not None:
        label = f"X={world_xy[0]:.2f} Y={world_xy[1]:.2f} m"
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return out


def locate_once(calib, detector="aruco", hsv_low=None, hsv_high=None):
    """Grab a frame, run the chosen detector, map to floor coords.
    Returns None if nothing was detected, else a dict with uv/x/y/frame."""
    frame = get_frame()
    check_frame_size(frame, calib)

    if detector == "aruco":
        uv = detect_aruco(frame)
    elif detector == "color":
        uv = detect_color_blob(frame, hsv_low, hsv_high)
    else:
        raise ValueError(f"Unknown detector: {detector}")

    if uv is None:
        return None

    x, y = image_to_world(uv, calib["H"])
    return {"uv": uv, "x": x, "y": y, "frame": frame}


def main():
    detector = sys.argv[1] if len(sys.argv) > 1 else "aruco"
    calib = load_calibration()
    print(f"Loaded calibration (rmse={calib['rmse']:.4f} m)")

    try:
        while True:
            result = locate_once(calib, detector=detector)
            if result is None:
                print("No object detected")
            else:
                x, y = result["x"], result["y"]
                dist = (x ** 2 + y ** 2) ** 0.5
                bearing = np.degrees(np.arctan2(y, x))
                print(f"Object at X={x:.2f} Y={y:.2f} m | {dist:.2f} m, {bearing:.1f} deg left")
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        release_frame_source()


if __name__ == "__main__":
    main()
