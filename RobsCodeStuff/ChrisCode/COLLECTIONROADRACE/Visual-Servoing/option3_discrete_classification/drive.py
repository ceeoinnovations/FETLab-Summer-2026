"""
Autonomous drive using discrete classification.

No centroid, no bounding box, no continuous controller anywhere in this
file. Each frame, the trained classifier (classifier_model.py) sorts the
image into one of config.CATEGORIES, and that category is looked up
directly in config.CATEGORY_MOTOR_COMMANDS to get a fixed (left, right)
motor speed pair — no steering/speed formula is computed at all. This is
the most structurally different of the three drive.py files: Options 1
and 2 both still hand off to a proportional controller, this one doesn't
have one.

A single flickered misclassification is smoothed out with a majority
vote over the last few frames (config.CATEGORY_VOTE_WINDOW) rather than
acting on every raw per-frame prediction.

Press Q to quit.
"""

import cv2
import time
import torch
from collections import deque, Counter
from torchvision import transforms
from PIL import Image
from config import (
    CAMERA, SERIAL, SERIAL_COLOR_SENSOR, CATEGORIES, CATEGORY_MOTOR_COMMANDS,
    CATEGORY_VOTE_WINDOW, PERCEPTION_MODEL_PATH, IMG_SIZE,
    OBSTACLE_REFLECTION_THRESHOLD, AVOID_BACKUP_SPEED, AVOID_BACKUP_TIME, AVOID_TURN_DEGREES,
    AVOID_DRIVE_SPEED, AVOID_DRIVE_TIME, apply_deadzone,
)
from classifier_model import build_classifier_model
from lelib import doubleMotor, colorSensor


def avoid_obstacle(dm):
    """Interrupt handler: something is right in front of the color sensor.
    Same blind, hardcoded escape maneuver as the other two options — this
    reflex doesn't go through the classifier at all."""
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
    print("Obstacle cleared — resuming.")


device = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading classifier on: {device}")
checkpoint = torch.load(PERCEPTION_MODEL_PATH, map_location="cpu")
model = build_classifier_model().to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_category(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        idx = int(logits.argmax(dim=1)[0])
    return CATEGORIES[idx]


dm = doubleMotor()
print("Connecting to motors...")
dm.connect(SERIAL)
print("Connected.\n")

cs = colorSensor()
print("Connecting to color sensor (used as a proximity sensor)...")
cs.connect(SERIAL_COLOR_SENSOR)
print("Connected.\n")

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    dm.stop()
    raise RuntimeError(f"Could not open camera: {CAMERA}\n"
                       "Check Camo Studio is running and connected.")
print("Camera connected. Driving autonomously. Press Q to quit.\n")

recent_categories = deque(maxlen=CATEGORY_VOTE_WINDOW)

try:
    while cap.isOpened():
        if cs.reflection() > OBSTACLE_REFLECTION_THRESHOLD:
            avoid_obstacle(dm)
            recent_categories.clear()
            continue

        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        raw_category = predict_category(frame)
        recent_categories.append(raw_category)
        # Majority vote over the last few frames — see config.CATEGORY_VOTE_WINDOW
        category = Counter(recent_categories).most_common(1)[0][0]

        left_speed, right_speed = CATEGORY_MOTOR_COMMANDS[category]
        dm.movement_move_tank(apply_deadzone(left_speed), apply_deadzone(right_speed))

        hud = frame.copy()
        cv2.putText(hud, f"{category}  (raw: {raw_category})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
        cv2.putText(hud, f"L: {left_speed:+4d}  R: {right_speed:+4d}",
                    (10, hud.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
        cv2.imshow("Autonomous Drive (classification)", hud)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.033)

finally:
    dm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
