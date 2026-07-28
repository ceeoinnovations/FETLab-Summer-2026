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
    accurately). Once stopped at the target, hold still for a good
    1-2 seconds before moving again — those "truly arrived and
    stationary" frames are exactly what teaches the model to stop
    cleanly instead of coasting into the object.
  - Vary how far off-center the object starts (not just distance) —
    steep approach angles are what teach correct turning direction.

Whenever you physically pick up the car or the target to start a new
drive, press N first (car should be stopped). This writes a new
session_id into labels.csv so train.py can split train/validation by
drive instead of by individual frame, and so nobody has to guess
where one drive ends and the next begins from the images alone.

Press Q to stop collection at any time.
"""

import cv2
import csv
import time
import statistics
from collections import deque
from pathlib import Path
from config import (
    CAMERA, SERIAL, CAPTURE_HZ, MOTOR_SPEED_MIN, MOTOR_SPEED_MAX, MOTOR_DIVIDE_BY,
    SMOOTH_MEDIAN_WINDOW, SMOOTH_EMA_ALPHA, apply_deadzone,
)
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

print(f"Connected. Capturing at {CAPTURE_HZ} Hz. Drive toward the object.")
print("Press N when you reposition the car/target for a new drive. Press Q to stop.\n")

frame_interval = 1.0 / CAPTURE_HZ
existing = sorted(IMG_DIR.glob("*.jpg"))
frame_idx = len(existing)

# Smoothing state: a short history for the median filter, plus a running EMA
# value carried across frames for each side.
left_history = deque(maxlen=SMOOTH_MEDIAN_WINDOW)
right_history = deque(maxlen=SMOOTH_MEDIAN_WINDOW)
ema_left = 0.0
ema_right = 0.0


def smooth(raw_value, history, ema_prev):
    """Median-filter then exponentially smooth a raw joystick reading.

    Returns (smoothed_value, updated_ema). The median filter rejects a
    single-frame spike/glitch (needs the majority of recent samples to
    agree before a big value is believed); the EMA then softens the
    remaining frame-to-frame jitter from natural hand tremor.
    """
    history.append(raw_value)
    median_value = statistics.median(history)
    ema_new = SMOOTH_EMA_ALPHA * ema_prev + (1 - SMOOTH_EMA_ALPHA) * median_value
    return ema_new

# Figure out what session_id to continue from if labels.csv already has rows.
# Restarting this script counts as a new session even if you forget to
# press N, since the car/target was necessarily disturbed in between runs.
session_id = 0
if LABELS_CSV.exists():
    with open(LABELS_CSV) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        last_row = None
        for row in reader:
            last_row = row
        if last_row is not None and len(last_row) >= 4:
            session_id = int(last_row[3]) + 1
            print(f"Resuming: this run starts at session_id={session_id}.")

with open(LABELS_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if frame_idx == 0:
        writer.writerow(["filename", "left_speed", "right_speed", "session_id"])

    last_capture = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera feed.")
                break

            # left_position/right_position return joystick percent (-100..100).
            # Smooth before dividing/rounding so a single-frame spike or hand
            # tremor doesn't turn into a lurch — see SMOOTH_* in config.py.
            raw_left = js.left_position() / MOTOR_DIVIDE_BY
            raw_right = js.right_position() / MOTOR_DIVIDE_BY
            ema_left = smooth(raw_left, left_history, ema_left)
            ema_right = smooth(raw_right, right_history, ema_right)
            left_speed, right_speed = round(ema_left), round(ema_right)

            now = time.time()
            if now - last_capture >= frame_interval:
                last_capture = now
                fname = f"{frame_idx:05d}.jpg"
                cv2.imwrite(str(IMG_DIR / fname), frame)
                writer.writerow([fname, left_speed, right_speed, session_id])
                f.flush()
                frame_idx += 1

            # mirror to motors so the human sees live feedback while driving
            # (deadzone applied here on top of the smoothing above)
            #
            # IMPORTANT: this must send the SAME numeric command (same units, same
            # scale) that drive.py will send for an identical predicted value.
            # Previously this divided by 10 while drive.py multiplied by 3 — a
            # 30x mismatch — so the autonomous car drove 30x harder than what was
            # ever demonstrated for a given label, which is why it couldn't stop
            # cleanly and pushed the target. Keep this line and the corresponding
            # line in drive.py identical (both just `apply_deadzone(speed)`).
            dm.movement_move_tank(apply_deadzone(left_speed), apply_deadzone(right_speed))

            hud = frame.copy()
            cv2.putText(hud, f"L: {left_speed:+4d}  R: {right_speed:+4d}  saved: {frame_idx}  session: {session_id}",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
            cv2.putText(hud, "N = new drive   Q = stop", (10, hud.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.imshow("Collect Data (driving)", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                dm.stop()
                session_id += 1
                print(f"New drive marked -> session_id={session_id}. "
                      "Reposition the car/target, then continue driving.")
    finally:
        dm.stop()
        cap.release()
        cv2.destroyAllWindows()

print(f"\nCollection complete. {frame_idx} samples saved to {DATA_DIR} across {session_id + 1} session(s).")

