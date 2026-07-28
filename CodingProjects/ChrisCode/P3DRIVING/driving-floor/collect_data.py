"""
Step 1 — Collect training images from the phone camera.

Mount the phone on the robot pointing downward, then hold each sketch
card under the camera one at a time and capture frames.

Controls:
  1-5   select symbol class
  Space save current frame
  Q     quit
"""

import cv2
import os
from config import CAMERA
from model import SYMBOL_CLASSES
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
for cls in SYMBOL_CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

counts  = {cls: len(os.listdir(os.path.join(DATA_DIR, cls))) for cls in SYMBOL_CLASSES}
KEY_MAP = {ord(str(i + 1)): SYMBOL_CLASSES[i] for i in range(len(SYMBOL_CLASSES))}

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera: {CAMERA}\n"
                       "Check the IP address in config.py and that the app is running.")

current_class = None

print("Keys: 1-5 select symbol | Space = capture | Q = quit")
for i, cls in enumerate(SYMBOL_CLASSES):
    print(f"  {i + 1} = {cls}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lost camera feed — check Wi-Fi connection.")
        break

    label = current_class or "— press 1-5 to select"
    cv2.putText(frame, f"Symbol: {label}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
    for i, cls in enumerate(SYMBOL_CLASSES):
        cv2.putText(frame, f"  {i + 1}: {cls}  ({counts[cls]} saved)",
                    (10, 62 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "Space = capture  |  Q = quit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Collect Data (driving)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in KEY_MAP:
        current_class = KEY_MAP[key]
        print(f"Selected: {current_class}")
    elif key == ord(' ') and current_class:
        path = os.path.join(DATA_DIR, current_class, f"{counts[current_class]:05d}.jpg")
        cv2.imwrite(path, frame)
        counts[current_class] += 1
        print(f"Saved {path}")

cap.release()
cv2.destroyAllWindows()
print("\nCollection complete. Image counts:", counts)
