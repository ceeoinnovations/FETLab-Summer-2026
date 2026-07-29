"""
Step 3 — Drive the double motor with your pose.

Loads the model trained by train.py, classifies your pose in real time,
and maps the predicted gesture to left/right motor speeds.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import torch
from torchvision import transforms
from model import build_model
from lelib import doubleMotor

SERIAL     = 2279       # change to your Bluetooth card serial number
MODEL_PATH = "pose_model.pt"

MOTOR_MAP = {
    "stop":       (   0,    0),
    "forward":    ( 100,  100),
    "backward":   (-100, -100),
    "turn_left":  ( -60,  100),
    "turn_right": ( 100,  -60),
}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
classes    = checkpoint["classes"]
device     = ("mps"  if torch.backends.mps.is_available()  else
              "cuda" if torch.cuda.is_available()           else "cpu")
model = build_model(num_classes=len(classes)).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Model loaded. Classes: {classes}")

dm = doubleMotor()
print("Connecting to double motor...")
dm.connect(SERIAL)
print("Connected. Strike a pose to drive. Press Q to quit.\n")

cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu()

        idx   = int(probs.argmax())
        label = classes[idx]
        conf  = float(probs[idx])

        left_speed, right_speed = MOTOR_MAP.get(label, (0, 0))
        dm.movement_move_tank(left_speed, right_speed)

        h = frame.shape[0]
        cv2.putText(frame, f"{label}  {conf:.0%}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 80), 2)
        cv2.putText(frame, f"L: {left_speed:+4d}  R: {right_speed:+4d}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)

        for i, cls in enumerate(classes):
            bar = int(float(probs[i]) * 180)
            y0, y1 = h - 14 - i * 20, h - 2 - i * 20
            cv2.rectangle(frame, (10, y0), (10 + bar, y1), (80, 200, 80), -1)
            cv2.putText(frame, cls, (200, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        cv2.imshow("Pose Drive (scratch)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.033)  # ~30 Hz

finally:
    dm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
