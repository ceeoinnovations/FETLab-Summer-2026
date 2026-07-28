"""
Step 1 - Collect centering data for the pan/yaw tracking rig.

Hardware:
  - A single LEGO motor rotates a camera mount in yaw (positive speed =
    clockwise, negative = counterclockwise).
  - A LEGO controller is mounted alongside it purely as a "force gauge":
    a shaft extends from its right joystick, and when that shaft contacts
    the target object, the joystick reading rises. This controller is
    NEVER used for driving - it's a training-only sensor.

Per-trial loop (one row per direction is drawn from --iterations, shuffled):
  1. APPROACH  - motor spins at a constant speed toward the target.
  2. CONTACT   - once the force-gauge reading crosses CONTACT_THRESHOLD,
                 the motor starts decelerating; harder contact = faster
                 deceleration (see config.MIN_DECEL_RATE / DECEL_GAIN).
  3. STOPPED   - once speed decays below STOP_SPEED_EPSILON, the motor is
                 commanded to 0 and the trial's data collection ends.
  4. REPOSITION - motor drives back the opposite direction for a random
                 duration, landing on a new random starting position
                 before the next trial begins. Nothing is recorded here.

Each recorded row is (image filename, signed motor speed) - a single
scalar label, since deployment is a single motor. Recording is active
only during APPROACH and CONTACT (i.e. "while the motor is moving"),
matching the spec exactly.

Controls:
  SPACE  e-stop / resume. Immediately halts the motor and freezes
         whatever phase you were in; pressing SPACE again resumes that
         same phase from where it left off.
  Q      quit entirely (also stops the motor).

Usage:
  python collect_data.py --iterations 20
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2

from config import (
    CAMERA, SERIAL, CAPTURE_HZ,
    APPROACH_SPEED, CONTACT_THRESHOLD,
    MIN_DECEL_RATE, DECEL_GAIN, STOP_SPEED_EPSILON,
    REPOSITION_SPEED, REPOSITION_MIN_SEC, REPOSITION_MAX_SEC,
)
from lelib import singleMotor, controller

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LABELS_CSV = DATA_DIR / "labels.csv"

parser = argparse.ArgumentParser(description="Collect pan/yaw centering data.")
parser.add_argument("--iterations", type=int, default=20,
                     help="Total number of trials to run, split as evenly as "
                          "possible between clockwise and counterclockwise "
                          "approaches, then shuffled.")
args = parser.parse_args()

n_cw = args.iterations // 2
n_ccw = args.iterations - n_cw
directions = [1] * n_cw + [-1] * n_ccw   # +1 = clockwise, -1 = counterclockwise
random.shuffle(directions)

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    raise RuntimeError(f"Could not open camera: {CAMERA}\n"
                       "Check Camo Studio is running and connected.")

print("Connecting to motor + force-gauge controller...")
motor = singleMotor()
motor.connect(SERIAL)
gauge = controller()
gauge.connect(SERIAL)

frame_interval = 1.0 / CAPTURE_HZ
existing = sorted(IMG_DIR.glob("*.jpg"))
frame_idx = len(existing)

print(f"Connected. Running {len(directions)} trials "
      f"({n_cw} clockwise, {n_ccw} counterclockwise).")
print("SPACE = e-stop/resume   Q = quit\n")

with open(LABELS_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if frame_idx == 0:
        writer.writerow(["filename", "motor_speed"])

    trial_idx = 0
    state = "approach"          # "approach" | "contact" | "reposition" | "done"
    paused = False
    prev_state = None           # state to restore on resume
    last_capture = 0.0
    last_loop = time.time()

    direction = directions[0]
    speed = APPROACH_SPEED * direction     # signed speed currently commanded
    reposition_remaining = 0.0              # countdown timer while repositioning
    motor.run(int(round(speed)))

    try:
        while trial_idx < len(directions):
            ret, frame = cap.read()
            if not ret:
                print("Lost camera feed.")
                break

            now = time.time()
            dt = max(0.0, now - last_loop)
            last_loop = now

            force = abs(gauge.right_position())

            if not paused:
                if state == "approach":
                    if force > CONTACT_THRESHOLD:
                        state = "contact"

                elif state == "contact":
                    decel_rate = MIN_DECEL_RATE + DECEL_GAIN * force
                    mag = max(0.0, abs(speed) - decel_rate * dt)
                    speed = mag * direction
                    motor.run(int(round(speed)))
                    if mag < STOP_SPEED_EPSILON:
                        speed = 0.0
                        motor.stop()
                        state = "reposition"
                        reposition_remaining = random.uniform(
                            REPOSITION_MIN_SEC, REPOSITION_MAX_SEC)
                        motor.run(int(-direction * REPOSITION_SPEED))

                elif state == "reposition":
                    reposition_remaining -= dt
                    if reposition_remaining <= 0:
                        motor.stop()
                        trial_idx += 1
                        if trial_idx < len(directions):
                            direction = directions[trial_idx]
                            speed = APPROACH_SPEED * direction
                            state = "approach"
                            motor.run(int(round(speed)))
                        else:
                            state = "done"

                # record while (and only while) the motor is actively
                # moving toward the target: approach + contact/decel
                if state in ("approach", "contact"):
                    if now - last_capture >= frame_interval:
                        last_capture = now
                        fname = f"{frame_idx:05d}.jpg"
                        cv2.imwrite(str(IMG_DIR / fname), frame)
                        writer.writerow([fname, int(round(speed))])
                        f.flush()
                        frame_idx += 1
                    # also log the final zero-speed frame right as it stops,
                    # so the model sees an explicit "you've arrived" sample
                    if state == "contact" and speed == 0.0:
                        fname = f"{frame_idx:05d}.jpg"
                        cv2.imwrite(str(IMG_DIR / fname), frame)
                        writer.writerow([fname, 0])
                        f.flush()
                        frame_idx += 1

            if state == "done":
                break

            hud = frame.copy()
            cv2.putText(hud, f"trial {trial_idx + 1}/{len(directions)}  "
                              f"dir: {'CW' if direction > 0 else 'CCW'}  "
                              f"speed: {speed:+5.1f}",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 80), 2)
            cv2.putText(hud, f"state: {state.upper()}  force: {force:3d}  saved: {frame_idx}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 220), 1)
            if paused:
                cv2.putText(hud, "E-STOPPED - press SPACE to resume",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(hud, "SPACE=e-stop/resume  Q=quit", (10, hud.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.imshow("Collect Data (pan/yaw)", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if not paused:
                    paused = True
                    motor.stop()
                    print(f"E-STOP. Trial {trial_idx + 1}, state={state}. "
                          "Press SPACE to resume.")
                else:
                    paused = False
                    # re-issue whatever command the current state implies
                    if state in ("approach", "contact"):
                        motor.run(int(round(speed)))
                    elif state == "reposition":
                        motor.run(int(-direction * REPOSITION_SPEED))
                    print("Resumed.")
    finally:
        motor.stop()
        cap.release()
        cv2.destroyAllWindows()

print(f"\nCollection complete. {frame_idx} samples saved to {DATA_DIR}")
