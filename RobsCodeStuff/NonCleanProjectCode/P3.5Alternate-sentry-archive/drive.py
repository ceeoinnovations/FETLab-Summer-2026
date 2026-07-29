"""
Step 3 - Run the trained model (deployment).

Streams the camera feed, feeds each frame through the trained
regression model, and sends the predicted motor speed straight to the
yaw motor - no thresholding beyond a small deadband, since output is
continuous. The force-gauge controller from data collection is not used
here at all; this is pure end-to-end image -> speed.

Press Q to stop.
"""

import cv2
import time
import torch
from torchvision import transforms
from config import CAMERA, SERIAL, MOTOR_SPEED_MIN, MOTOR_SPEED_MAX, IMG_SIZE, RUN_DEADBAND
from model import build_model
from lelib import singleMotor

MODEL_PATH = "pan_model.pt"


def denormalize_speed(v):
    mid = (MOTOR_SPEED_MAX + MOTOR_SPEED_MIN) / 2
    span = (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN) / 2
    return v * span + mid


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
device = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
model = build_model().to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print("Model loaded.")

motor = singleMotor()
print("Connecting to motor...")
motor.connect(SERIAL)
print("Connected.\n")

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    motor.stop()
    raise RuntimeError(f"Could not open camera: {CAMERA}\n"
                       "Check Camo Studio is running and connected.")
print("Camera connected. Running autonomously. Press Q to quit.\n")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor)[0].cpu()

        speed = denormalize_speed(float(pred[0]))

        if abs(speed) < RUN_DEADBAND:
            motor.stop()
            speed = 0.0
        else:
            motor.run(int(round(speed)))

        cv2.putText(frame, f"speed: {speed:+5.1f}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
        cv2.imshow("Autonomous Pan/Yaw", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.033)  # ~30 Hz inference, independent of the 10 Hz training capture rate

finally:
    motor.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
