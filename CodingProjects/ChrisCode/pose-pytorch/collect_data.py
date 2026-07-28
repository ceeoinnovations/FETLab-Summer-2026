"""
Step 1 — Collect training images.

Controls:
  1-5   select gesture class
  Space save current frame to that class folder
  Q     quit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import os
from model import GESTURE_CLASSES
from camlib import pick_camera

DATA_DIR = Path(__file__).parent / "data"
for cls in GESTURE_CLASSES:
    (DATA_DIR / cls).mkdir(parents=True, exist_ok=True)

counts  = {cls: len(list((DATA_DIR / cls).iterdir())) for cls in GESTURE_CLASSES}
KEY_MAP = {ord(str(i + 1)): GESTURE_CLASSES[i] for i in range(len(GESTURE_CLASSES))}

cap, _        = pick_camera()
current_class = None

print("Keys: 1-5 select class | Space = capture | Q = quit")
for i, cls in enumerate(GESTURE_CLASSES):
    print(f"  {i + 1} = {cls}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    label = current_class or "— press 1-5 to select a class"
    cv2.putText(frame, f"Class: {label}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
    for i, cls in enumerate(GESTURE_CLASSES):
        cv2.putText(frame, f"  {i + 1}: {cls}  ({counts[cls]} saved)",
                    (10, 62 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, "Space = capture  |  Q = quit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key in KEY_MAP:
        current_class = KEY_MAP[key]
        print(f"Selected: {current_class}")
    elif key == ord(' ') and current_class:
        path = str(DATA_DIR / current_class / f"{counts[current_class]:05d}.jpg")
        cv2.imwrite(path, frame)
        counts[current_class] += 1
        print(f"Saved {path}")

cap.release()
cv2.destroyAllWindows()
print("\nCollection complete. Image counts:", counts)
