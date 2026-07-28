"""
Step 1 — Collect training images.

From-scratch training needs more data than transfer learning.
Aim for at least 100 images per class (vs 30-50 for pose-pytorch).
The counter turns green when you hit the target.

Controls:
  1-5   select gesture class
  Space save current frame to that class folder
  Q     quit
"""

import cv2
import os
from model import GESTURE_CLASSES

DATA_DIR = "data"
TARGET   = 100  # recommended minimum per class

for cls in GESTURE_CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)

counts  = {cls: len(os.listdir(os.path.join(DATA_DIR, cls))) for cls in GESTURE_CLASSES}
KEY_MAP = {ord(str(i + 1)): GESTURE_CLASSES[i] for i in range(len(GESTURE_CLASSES))}

cap           = cv2.VideoCapture(0)
current_class = None

print(f"Collect at least {TARGET} images per class for reliable from-scratch training.")
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
        count  = counts[cls]
        color  = (0, 220, 0) if count >= TARGET else (200, 200, 200)
        marker = "✓" if count >= TARGET else f"{count}/{TARGET}"
        cv2.putText(frame, f"  {i + 1}: {cls}  ({marker})",
                    (10, 62 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    cv2.putText(frame, "Space = capture  |  Q = quit",
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow("Collect Data (scratch)", frame)
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
