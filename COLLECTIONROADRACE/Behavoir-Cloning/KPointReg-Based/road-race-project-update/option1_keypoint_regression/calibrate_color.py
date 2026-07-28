"""
Interactive helper to find HSV_LOWER/HSV_UPPER for detect.py.

Lighting and target color vary a lot from room to room and camera to
camera — don't guess values by hand, use this instead. Shows the live feed
side-by-side with the resulting mask; adjust the six sliders until the
target is the only thing showing up white in the mask.

Controls:
  S = print the current HSV_LOWER/HSV_UPPER + area_frac reading (paste
      HSV_LOWER/HSV_UPPER into config.py; use the area_frac reading at your
      desired stopping distance to help set STOP_AREA_FRACTION there too)
  Q = quit
"""

import cv2
import numpy as np
from config import CAMERA
from detect import get_target_color

WINDOW = "calibrate_color  (S = print values, Q = quit)"


def _nothing(_):
    pass


cv2.namedWindow(WINDOW)
cv2.createTrackbar("H min", WINDOW, 15, 179, _nothing)
cv2.createTrackbar("H max", WINDOW, 40, 179, _nothing)
cv2.createTrackbar("S min", WINDOW, 80, 255, _nothing)
cv2.createTrackbar("S max", WINDOW, 255, 255, _nothing)
cv2.createTrackbar("V min", WINDOW, 80, 255, _nothing)
cv2.createTrackbar("V max", WINDOW, 255, 255, _nothing)

cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera: {CAMERA}\nCheck Camo Studio is running and connected.")

print("Point the camera at the target. Adjust sliders until ONLY the target")
print("shows up white in the right-hand mask panel (no background noise).")
print("Press S to print the values, Q to quit.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        lower = (
            cv2.getTrackbarPos("H min", WINDOW),
            cv2.getTrackbarPos("S min", WINDOW),
            cv2.getTrackbarPos("V min", WINDOW),
        )
        upper = (
            cv2.getTrackbarPos("H max", WINDOW),
            cv2.getTrackbarPos("S max", WINDOW),
            cv2.getTrackbarPos("V max", WINDOW),
        )

        target = get_target_color(frame, hsv_lower=lower, hsv_upper=upper)
        mask = target["mask"] if target is not None else cv2.inRange(
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower, upper)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        label = f"area_frac={target['area_frac']:.3f}" if target is not None else "NOT FOUND"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        h = frame.shape[0]
        mask_bgr = cv2.resize(mask_bgr, (frame.shape[1], h))
        combined = np.hstack([frame, mask_bgr])
        cv2.imshow(WINDOW, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            print(f"HSV_LOWER = {lower}")
            print(f"HSV_UPPER = {upper}")
            if target is not None:
                print(f"(current area_frac at this distance: {target['area_frac']:.3f} "
                      "— useful for setting FORWARD_SLOWDOWN_AREA/STOP_AREA_FRACTION)")
            print()
finally:
    cap.release()
    cv2.destroyAllWindows()
