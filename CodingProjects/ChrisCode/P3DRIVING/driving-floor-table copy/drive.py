"""
Step 3 — Read sketch commands and drive.

The phone camera (mounted pointing down) streams video to this script.
Each frame is classified by the trained model. If confidence exceeds
the threshold in config.py, the corresponding motor command runs.
Below threshold the motors stop — the robot won't act on a guess.

Place a sketch card under the robot to issue a command. Swap cards to
change direction. Remove the card (or show a blank/stop card) to stop.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import time
import torch
from torchvision import transforms
from config import CAMERA, SERIAL, CONFIDENCE_THRESHOLD
from model import build_model, MOTOR_MAP
from lelib import doubleMotor

MODEL_PATH = "symbol_model.pt"

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
print("Connected.\n")

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    dm.stop()
    raise RuntimeError(f"Could not open camera: {CAMERA}\n"
                       "Check the IP address in config.py and that the app is running.")
print("Camera connected. Place a sketch card under the robot. Press Q to quit.\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu()

        idx   = int(probs.argmax())
        label = classes[idx]
        conf  = float(probs[idx])

        if conf >= CONFIDENCE_THRESHOLD:
            left_speed, right_speed = MOTOR_MAP.get(label, (0, 0))
            status_color = (0, 255, 80)
        else:
            left_speed, right_speed = 0, 0
            label  = f"{label}?"   # mark uncertain predictions
            status_color = (0, 140, 255)

        dm.movement_move_tank(left_speed/3, right_speed/3)

        # ── HUD ──────────────────────────────────────────────────────────
        h = frame.shape[0]
        cv2.putText(frame, f"{label}  {conf:.0%}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)
        cv2.putText(frame, f"L: {left_speed:+4d}  R: {right_speed:+4d}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
        cv2.putText(frame, f"threshold: {CONFIDENCE_THRESHOLD:.0%}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        for i, cls in enumerate(classes):
            bar = int(float(probs[i]) * 180)
            y0, y1 = h - 34 - i * 20, h - 22 - i * 20
            cv2.rectangle(frame, (10, y0), (10 + bar, y1), (80, 200, 80), -1)
            cv2.putText(frame, cls, (200, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        cv2.imshow("Driving", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.033)  # ~30 Hz

finally:
    dm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
