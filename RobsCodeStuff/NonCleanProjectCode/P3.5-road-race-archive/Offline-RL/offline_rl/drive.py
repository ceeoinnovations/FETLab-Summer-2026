"""
Autonomous drive using the trained offline RL actor (TD3+BC).

detect.py finds the target's compact state (cx_norm, cy_norm, area_frac,
visible) each frame via color thresholding — same detector that generated
this actor's training labels. The actor maps that state directly to
[left_speed, right_speed]; there's no hardcoded control law here, unlike
the other option* projects.

The actor was trained almost entirely on frames where the target WAS
visible (that's what human demonstrations mostly look like), so it has
little to no experience with "target not visible." If the target isn't
found at all, this falls back to a hardcoded spin-and-search reflex
instead of trusting the actor outside its training distribution.

Press Q to quit.
"""

import cv2
import time
import csv
import statistics
from collections import deque
from pathlib import Path

import torch

from config import (
    CAMERA, SERIAL, SERIAL_COLOR_SENSOR,
    SEARCH_TURN_SPEED, SEARCH_REVERSE_DIRECTION_AFTER,
    DETECT_MEDIAN_WINDOW, DETECT_EMA_ALPHA,
    OBSTACLE_REFLECTION_THRESHOLD, AVOID_BACKUP_SPEED, AVOID_BACKUP_TIME, AVOID_TURN_DEGREES,
    AVOID_DRIVE_SPEED, AVOID_DRIVE_TIME, apply_deadzone,
    MOTOR_SPEED_MIN, MOTOR_SPEED_MAX,
    EXPORT_EXPERT_DATA, EXPERT_DATA_DIR, EXPERT_CAPTURE_HZ,
)
from detect import get_target_color, draw_debug
from lelib import doubleMotor, colorSensor
from offline_rl_actor_model import build_offline_rl_actor

MODEL_PATH = "offline_rl_actor.pt"
DEVICE = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")


def denormalize_speed(v):
    mid = (MOTOR_SPEED_MAX + MOTOR_SPEED_MIN) / 2
    span = (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN) / 2
    return v * span + mid


def avoid_obstacle(dm):
    """Interrupt handler: something is right in front of the color sensor.
    Blind, hardcoded escape maneuver — a reflex, not a decision."""
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
    print("Obstacle cleared — resuming search.")


def smooth(raw_value, history, ema_prev, ema_alpha=DETECT_EMA_ALPHA):
    history.append(raw_value)
    median_value = statistics.median(history)
    return ema_alpha * ema_prev + (1 - ema_alpha) * median_value


print(f"Loading actor from {MODEL_PATH}...")
if not Path(MODEL_PATH).exists():
    raise SystemExit(f"Could not find {MODEL_PATH} — run train_offline_rl.py first.")
actor = build_offline_rl_actor().to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
actor.load_state_dict(checkpoint["state_dict"])
actor.eval()
print("Actor loaded.\n")

dm = doubleMotor()
print("Connecting to motors...")
dm.connect(SERIAL)

cs = colorSensor()
print("Connecting to color sensor...")
cs.connect(SERIAL_COLOR_SENSOR)

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    dm.stop()
    raise RuntimeError(f"Could not open camera: {CAMERA}\nCheck Camo Studio is running and connected.")
print("Camera connected. Driving autonomously. Press Q to quit.\n")

# ── Optional: export this run as training data for OTHER projects ──────────
# See config.EXPORT_EXPERT_DATA's comment — this is unlikely to usefully
# improve THIS model if fed back into train_offline_rl.py (the actor is a
# deterministic function of its input, same contrast problem as any other
# autonomous driving), but is kept for consistency with the rest of this
# project family and is still useful data for other model types.
expert_writer = expert_csv_file = None
expert_frame_idx = 0
expert_session_id = 0
last_expert_capture = 0.0
expert_capture_interval = 1.0 / EXPERT_CAPTURE_HZ

if EXPORT_EXPERT_DATA:
    expert_dir = Path(__file__).parent / EXPERT_DATA_DIR
    expert_img_dir = expert_dir / "images"
    expert_img_dir.mkdir(parents=True, exist_ok=True)
    expert_labels_csv = expert_dir / "labels.csv"

    existing = sorted(expert_img_dir.glob("*.jpg"))
    expert_frame_idx = len(existing)

    if expert_labels_csv.exists():
        with open(expert_labels_csv) as f:
            reader = csv.reader(f)
            next(reader, None)
            last_row = None
            for row in reader:
                last_row = row
            if last_row is not None and len(last_row) >= 4:
                expert_session_id = int(last_row[3]) + 1

    expert_csv_file = open(expert_labels_csv, "a", newline="")
    expert_writer = csv.writer(expert_csv_file)
    if expert_frame_idx == 0:
        expert_writer.writerow(["filename", "left_speed", "right_speed", "session_id"])
    print(f"Exporting this run to {expert_dir} at {EXPERT_CAPTURE_HZ} Hz "
          f"(session_id={expert_session_id}).\n")

cx_history = deque(maxlen=DETECT_MEDIAN_WINDOW)
cy_history = deque(maxlen=DETECT_MEDIAN_WINDOW)
area_history = deque(maxlen=DETECT_MEDIAN_WINDOW)
ema_cx = ema_cy = ema_area = 0.0
last_turn_sign = 1
search_started_at = None
search_direction = 1

try:
    while cap.isOpened():
        if cs.reflection() > OBSTACLE_REFLECTION_THRESHOLD:
            avoid_obstacle(dm)
            cx_history.clear(); cy_history.clear(); area_history.clear()
            search_started_at = None
            continue

        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        target = get_target_color(frame)

        if target is not None:
            search_started_at = None
            ema_cx = smooth(target["cx_norm"], cx_history, ema_cx)
            ema_cy = smooth(target["cy_norm"], cy_history, ema_cy)
            ema_area = smooth(target["area_frac"], area_history, ema_area)

            s = torch.tensor([[ema_cx, ema_cy, ema_area, 1.0]], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                action = actor(s)[0].cpu()
            left_speed = denormalize_speed(float(action[0]))
            right_speed = denormalize_speed(float(action[1]))
            last_turn_sign = 1 if ema_cx > 0 else -1
        else:
            if search_started_at is None:
                search_started_at = time.time()
                search_direction = last_turn_sign
            elif time.time() - search_started_at > SEARCH_REVERSE_DIRECTION_AFTER:
                search_direction *= -1
                search_started_at = time.time()
            left_speed = SEARCH_TURN_SPEED * search_direction
            right_speed = -SEARCH_TURN_SPEED * search_direction
            cx_history.clear(); cy_history.clear(); area_history.clear()

        left_speed, right_speed = apply_deadzone(left_speed), apply_deadzone(right_speed)

        if EXPORT_EXPERT_DATA:
            now = time.time()
            if now - last_expert_capture >= expert_capture_interval:
                last_expert_capture = now
                fname = f"{expert_frame_idx:05d}.jpg"
                cv2.imwrite(str(expert_img_dir / fname), frame)
                expert_writer.writerow([fname, round(left_speed), round(right_speed), expert_session_id])
                expert_csv_file.flush()
                expert_frame_idx += 1

        dm.movement_move_tank(left_speed, right_speed)

        hud = draw_debug(frame, target)
        cv2.putText(hud, f"L: {left_speed:+5.1f}  R: {right_speed:+5.1f}",
                    (10, hud.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
        cv2.imshow("Offline RL - Drive", hud)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        time.sleep(0.033)

finally:
    dm.stop()
    cap.release()
    cv2.destroyAllWindows()
    if expert_csv_file is not None:
        expert_csv_file.close()
        print(f"Expert data export: {expert_frame_idx} frames saved to "
              f"{Path(__file__).parent / EXPERT_DATA_DIR}")
    print("Stopped.")
