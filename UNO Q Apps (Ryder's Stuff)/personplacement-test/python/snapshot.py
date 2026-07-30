"""Grab one frame from the USB webcam and save it as an image.

Use this to get a still image you can open in an image viewer/editor to
read off the (u,v) pixel coordinates of your 6 calibration points for
points.csv.

    cd ~/ArduinoApps/personplacement-test/python
    python3 snapshot.py
    python3 snapshot.py my_frame.jpg   # custom output path
"""
import sys

import cv2

CAM_INDEX = 0  # /dev/video0 -- USB webcam


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "snapshot.jpg"

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAM_INDEX}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Frame grab failed")

    h, w = frame.shape[:2]
    cv2.imwrite(out_path, frame)
    print(f"Saved {out_path} ({w}x{h})")


if __name__ == "__main__":
    main()
