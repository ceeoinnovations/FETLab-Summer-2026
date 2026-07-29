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
    accurately)
.
Press SPACE to pause/resume RECORDING at any time (resuming triggers a
3-second countdown before frames start being saved again). Press Q to
stop the script entirely. The car remains fully drivable in every state -
pausing only stops writing (image, speed) pairs to disk, it does not
stop the motors.

Live driving controls:
  =/-  raise/lower the SPEED SCALE (a multiplier applied to the joystick
       command before it reaches the motors - handy for slowing the whole
       car down for the close-approach runs).
  ]/[  raise/lower the RAMP RATE (how fast the motors are allowed to
       accelerate, in units/second). Lower ramp rate = smoother, slower
       acceleration instead of instant joystick->motor jumps. This only
       affects speeding up - releasing the joystick back to center always
       brings the car to a stop quickly (after a brief debounce so BLE
       jitter doesn't cause false stops), regardless of the ramp rate.
The recorded label always matches the actual (scaled + ramped) speed sent
to the motors, so the model learns the smoothed behavior, not the raw
joystick input.
"""

import cv2
import csv
import time
from pathlib import Path
from config import CAMERA, SERIAL, CAPTURE_HZ, MOTOR_SPEED_MIN, MOTOR_SPEED_MAX
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
print("SPACE = pause/resume   =/- = speed scale   ]/[ = ramp rate   Q = stop\n")

frame_interval = 1.0 / CAPTURE_HZ
existing = sorted(IMG_DIR.glob("*.jpg"))
frame_idx = len(existing)

with open(LABELS_CSV, "a", newline="") as f:
    writer = csv.writer(f)
    if frame_idx == 0:
        writer.writerow(["filename", "left_speed", "right_speed"])

    COUNTDOWN_SECS = 3.0

    # ── Speed scale: multiplies the joystick command before it hits the motors ──
    SPEED_SCALE_MIN, SPEED_SCALE_MAX, SPEED_SCALE_STEP = 0.1, 1.0, 0.05
    speed_scale = 1.0

    # ── Ramp rate: max ACCELERATION, in speed-units/second (slew-rate limit) ──
    RAMP_RATE_MIN, RAMP_RATE_MAX, RAMP_RATE_STEP = 50.0, 2000.0, 50.0
    ramp_rate = 400.0  # units/sec; ~0.25s to go 0->100

    # ── Quick-stop: once the joystick has been centered for ZERO_HOLD_SECS,
    # snap the motors down to 0 at STOP_RATE (much faster than ramp_rate) so
    # the car doesn't coast/drift after you let go. The short hold prevents a
    # single noisy zero reading over BLE from causing a false stop mid-drive.
    ZERO_HOLD_SECS = 0.1
    STOP_RATE = 2000.0  # units/sec

    def ramp_toward(current, target, max_delta):
        if target > current:
            return min(target, current + max_delta)
        else:
            return max(target, current - max_delta)

    last_capture = 0.0
    last_loop = time.time()
    current_left, current_right = 0.0, 0.0   # actual (ramped) speeds sent to motors
    zero_since = None                        # time joystick was first seen centered
    state = "running"          # "running" | "paused" | "countdown" (recording status only)
    countdown_start = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lost camera feed.")
                break

            # left_position/right_position return joystick percent (-100..100).
            # These are the RAW joystick command; the actual motor command
            # (and recorded label) is the scaled + ramped version below.
            joy_left, joy_right = js.left_position(), js.right_position()

            now = time.time()
            dt = max(0.0, now - last_loop)
            last_loop = now

            # advance countdown -> running once time is up
            if state == "countdown" and (now - countdown_start) >= COUNTDOWN_SECS:
                state = "running"
                last_capture = now  # avoid an instant capture on resume

            # the car is drivable in every state - only RECORDING is gated by `state`
            target_left = joy_left * speed_scale
            target_right = joy_right * speed_scale

            # debounce: track how long the joystick has been centered
            if joy_left == 0 and joy_right == 0:
                if zero_since is None:
                    zero_since = now
                held_zero = (now - zero_since) >= ZERO_HOLD_SECS
            else:
                zero_since = None
                held_zero = False

            # once confirmed centered, snap to a stop quickly instead of
            # coasting down at the (possibly slow) acceleration ramp_rate
            active_rate = STOP_RATE if held_zero else ramp_rate

            max_delta = active_rate * dt
            current_left = ramp_toward(current_left, target_left, max_delta)
            current_right = ramp_toward(current_right, target_right, max_delta)

            dm.movement_move_tank(current_left, current_right)

            if state == "running":
                if now - last_capture >= frame_interval:
                    last_capture = now
                    fname = f"{frame_idx:05d}.jpg"
                    cv2.imwrite(str(IMG_DIR / fname), frame)
                    # record the ACTUAL (scaled + ramped) speed, since that's
                    # what really happened for this frame
                    writer.writerow([fname, round(current_left, 1), round(current_right, 1)])
                    f.flush()
                    frame_idx += 1

            hud = frame.copy()
            cv2.putText(hud, f"L: {current_left:+5.1f}  R: {current_right:+5.1f}  saved: {frame_idx}",
                        (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
            cv2.putText(hud, f"speed scale: {speed_scale:.2f}x   ramp rate: {ramp_rate:.0f}/s",
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 220), 1)

            if state == "paused":
                cv2.putText(hud, "NOT RECORDING (drive freely) - SPACE to resume",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
            elif state == "countdown":
                remaining = COUNTDOWN_SECS - (now - countdown_start)
                count_text = str(int(remaining) + 1) if remaining > 0 else ""
                cv2.putText(hud, f"Recording resumes in {count_text}... (still drivable)",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.putText(hud, "SPACE=pause/resume  =/- scale  ]/[ ramp  Q=stop",
                        (10, hud.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            cv2.imshow("Collect Data (driving)", hud)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                if state == "running":
                    state = "paused"
                    print("Recording paused (still drivable). Press SPACE to resume (3s countdown).")
                elif state == "paused":
                    state = "countdown"
                    countdown_start = now
                # if a key is pressed mid-countdown, ignore it (let countdown finish)
            elif key in (ord('='), ord('+')):
                speed_scale = min(SPEED_SCALE_MAX, speed_scale + SPEED_SCALE_STEP)
                print(f"Speed scale: {speed_scale:.2f}x")
            elif key in (ord('-'), ord('_')):
                speed_scale = max(SPEED_SCALE_MIN, speed_scale - SPEED_SCALE_STEP)
                print(f"Speed scale: {speed_scale:.2f}x")
            elif key == ord(']'):
                ramp_rate = min(RAMP_RATE_MAX, ramp_rate + RAMP_RATE_STEP)
                print(f"Ramp rate: {ramp_rate:.0f}/s")
            elif key == ord('['):
                ramp_rate = max(RAMP_RATE_MIN, ramp_rate - RAMP_RATE_STEP)
                print(f"Ramp rate: {ramp_rate:.0f}/s")
    finally:
        dm.stop()
        cap.release()
        cv2.destroyAllWindows()

print(f"\nCollection complete. {frame_idx} samples saved to {DATA_DIR}")