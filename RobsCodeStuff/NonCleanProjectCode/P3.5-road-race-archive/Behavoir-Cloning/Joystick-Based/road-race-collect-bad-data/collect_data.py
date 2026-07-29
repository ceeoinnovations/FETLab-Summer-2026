"""
Step 1 — Collect behavior-cloning data.

A human drives the LEGO car toward the object using the LEGO joystick.
At a fixed rate (CAPTURE_HZ), this script saves the current camera
frame paired with the joystick's current left/right motor speed
command. This (image, speed) pair is the training label — no manual
classification needed.

Run several short drives:
  - Full approaches from varying distances/angles
  - Several runs that start CLOSE to the object and only cover the
    final slow-approach/stop phase (this region is underrepresented
    in normal full-length drives but matters most for stopping
    accurately)
.
Press Q to stop collection at any time.
"""

import cv2
import csv
import time
from pathlib import Path
from config import CAMERA, SERIAL, CAPTURE_HZ, MOTOR_SPEED_MIN, MOTOR_SPEED_MAX
from lelib import doubleMotor, controller

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LABELS_CSV = DATA_DIR / "labels.csv"

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera: {CAMERA}\n"
                       "Check Camo Studio is running and connected.")

print("Connecting to joystick + motors...")
dm = doubleMotor()
dm.connect(SERIAL)
js = controller()
js.connect(SERIAL)

print(f"Connected. Capturing at {CAPTURE_HZ} Hz. Drive toward the object. Press Q to stop.\n")

frame_interval = 1.0 / CAPTURE_HZ
existing = sorted(IMG_DIR.glob("*.jpg"))
frame_idx = len(existing)

with open(LABELS_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if frame_idx == 0:
        writer.writerow(["filename", "left_speed", "right_speed"])

    last_capture = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera feed.")
                break

            # left_position/right_position return joystick percent (-100..100),
            # which doubles directly as the target motor speed labels.
            left_speed, right_speed = round(js.left_position()/4), round(js.right_position()/4)

            now = time.time()
            if now - last_capture >= frame_interval:
                last_capture = now
                fname = f"{frame_idx:05d}.jpg"
                cv2.imwrite(str(IMG_DIR / fname), frame)
                writer.writerow([fname, left_speed, right_speed])
                f.flush()
                frame_idx += 1

            # mirror joystick to motors so the human sees live feedback while driving
            dm.movement_move_tank(left_speed, right_speed)

            hud = frame.copy()
            cv2.putText(hud, f"L: {left_speed:+4d}  R: {right_speed:+4d}  saved: {frame_idx}",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
            cv2.putText(hud, "Q = stop", (10, hud.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.imshow("Collect Data (driving)", hud)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        dm.stop()
        cap.release()
        cv2.destroyAllWindows()

print(f"\nCollection complete. {frame_idx} samples saved to {DATA_DIR}")
