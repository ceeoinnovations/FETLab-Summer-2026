"""
Step 3 — Autonomous drive using the trained behavior-cloning model.

Streams the phone camera feed (via Camo Studio), feeds each frame
through the trained regression model, and sends the predicted
left/right motor speeds straight to the LEGO hub — no thresholding,
since output is continuous (unlike the classifier-based drive.py).

Press Q to stop.
"""

import cv2
import time
import torch
from torchvision import transforms
from config import (
    CAMERA, SERIAL, SERIAL_COLOR_SENSOR, MOTOR_SPEED_MIN, MOTOR_SPEED_MAX, IMG_SIZE,
    OBSTACLE_REFLECTION_THRESHOLD, AVOID_BACKUP_SPEED, AVOID_BACKUP_TIME, AVOID_TURN_DEGREES,
    AVOID_DRIVE_SPEED, AVOID_DRIVE_TIME, apply_deadzone,
)
from model import build_model
from lelib import doubleMotor, colorSensor

MODEL_PATH = "drive_model.pt"


def denormalize_speed(v):
    mid = (MOTOR_SPEED_MAX + MOTOR_SPEED_MIN) / 2
    span = (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN) / 2
    return v * span + mid


def avoid_obstacle(dm):
    """Interrupt handler: something is right in front of the color sensor.

    Blind, hardcoded escape maneuver — turn away, drive past the obstacle,
    turn back, then hand control back to the caller so it resumes looking
    for the target as usual. This does not use the camera/model at all;
    it's a reflex, not a decision.
    """
    print("Obstacle detected near color sensor — avoiding.")
    dm.stop()
    dm.run(AVOID_BACKUP_SPEED)
    time.sleep(AVOID_BACKUP_TIME)
    dm.turn_left(AVOID_TURN_DEGREES)
    dm.run(AVOID_DRIVE_SPEED)
    time.sleep(AVOID_DRIVE_TIME)
    dm.stop()
    dm.turn_right(AVOID_TURN_DEGREES)
    dm.stop()
    print("Obstacle cleared — resuming target search.")


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

dm = doubleMotor()
print("Connecting to motors...")
dm.connect(SERIAL)
print("Connected.\n")

cs = colorSensor()
print("Connecting to color sensor (used as a proximity sensor)...")
cs.connect(SERIAL_COLOR_SENSOR)  # separate card from the motor — see config.py
print("Connected.\n")

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    dm.stop()
    raise RuntimeError(f"Could not open camera: {CAMERA}\n"
                       "Check Camo Studio is running and connected.")
print("Camera connected. Driving autonomously. Press Q to quit.\n")

try:
    while cap.isOpened():
        # ── Interrupt: obstacle check takes priority over the model every frame ──
        # This is checked before inference, so an object placed in front of the
        # car is handled immediately instead of waiting for the model to react
        # to it visually (which it was never trained to do).
        if cs.reflection() > OBSTACLE_REFLECTION_THRESHOLD:
            avoid_obstacle(dm)
            continue  # skip this frame's inference, resume the normal loop

        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor)[0].cpu()

        left_speed = denormalize_speed(float(pred[0]))
        right_speed = denormalize_speed(float(pred[1]))

        # IMPORTANT: this must match collect_data.py's motor command exactly for
        # a given speed value (same units, same scale) — that's what makes the
        # demonstrated behavior and the deployed behavior the same physical motion.
        # Previously this multiplied by 3 while collection divided by 10 (a 30x
        # mismatch), which made the car drive far harder than anything it was
        # shown, causing overshoot into the target. No extra multiplier here.
        dm.movement_move_tank(apply_deadzone(left_speed), apply_deadzone(right_speed))

        cv2.putText(frame, f"L: {left_speed:+5.1f}  R: {right_speed:+5.1f}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 80), 2)
        cv2.imshow("Autonomous Drive", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.033)  # ~30 Hz inference, independent of 10 Hz training capture rate

finally:
    dm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
