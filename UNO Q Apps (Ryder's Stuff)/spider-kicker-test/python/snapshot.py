"""Grab one cropped frame from the rig camera and save it as an image, for
reading off (u, v) calibration points into points.csv.

Uses the same arduino.app_peripherals.camera.Camera class and center_crop()
helper main.py uses at runtime (not a raw cv2.VideoCapture), at the same
resolution/crop size - this guarantees the saved image is pixel-for-pixel
what object_detection.detect() sees, so pixel coordinates read off it line up
exactly with what calibrate.py/main.py will use them for.

The app must be stopped first - only one process can hold the USB camera:

    arduino-app-cli app stop ~/ArduinoApps/spider-kicker-test
    cd ~/ArduinoApps/spider-kicker-test/python
    python3 snapshot.py
    python3 snapshot.py my_frame.jpg   # custom output path

CAMERA_RESOLUTION/MODEL_INPUT_SIZE below must be kept in sync with main.py's
own constants of the same name.
"""
import sys

import cv2

import imaging
from arduino.app_peripherals.camera import Camera

CAMERA_RESOLUTION = (640, 480)
MODEL_INPUT_SIZE = 480


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "snapshot.jpg"

    camera = Camera(resolution=CAMERA_RESOLUTION)
    camera.start()
    try:
        frame = camera.capture()
        if frame is None:
            raise RuntimeError("Frame grab failed")
    finally:
        camera.stop()

    frame = imaging.center_crop(frame, MODEL_INPUT_SIZE)
    h, w = frame.shape[:2]
    cv2.imwrite(out_path, frame)
    print(f"Saved {out_path} ({w}x{h})")


if __name__ == "__main__":
    main()
