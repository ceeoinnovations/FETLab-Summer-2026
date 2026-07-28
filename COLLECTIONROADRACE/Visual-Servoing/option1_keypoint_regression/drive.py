"""
Autonomous drive using bounding-box + centroid vision control.

No neural network here — detect.py finds the target's bounding box each
frame via color thresholding, and this loop steers toward its centroid and
slows down as its area grows (i.e. as it gets closer), stopping once it's
close enough. See detect.py's docstring for why this replaced the learned
steering model.

If the target isn't visible at all, the car spins in place to search for
it, favoring whichever direction it was last seen drifting toward.

Press Q to quit.
"""

import cv2
import time
import statistics
from collections import deque
from config import (
    CAMERA, SERIAL, SERIAL_COLOR_SENSOR,
    STEER_GAIN, STEER_MAX,
    FORWARD_MAX_SPEED, FORWARD_SLOWDOWN_AREA, STOP_AREA_FRACTION,
    SEARCH_TURN_SPEED, SEARCH_REVERSE_DIRECTION_AFTER,
    DETECT_MEDIAN_WINDOW, DETECT_EMA_ALPHA,
    OBSTACLE_REFLECTION_THRESHOLD, AVOID_BACKUP_SPEED, AVOID_BACKUP_TIME, AVOID_TURN_DEGREES,
    AVOID_DRIVE_SPEED, AVOID_DRIVE_TIME, apply_deadzone,
)
from detect import get_target, draw_debug
from lelib import doubleMotor, colorSensor


def avoid_obstacle(dm):
    """Interrupt handler: something is right in front of the color sensor.

    Blind, hardcoded escape maneuver — turn away, drive past the obstacle,
    turn back, then hand control back to the caller so it resumes looking
    for the target as usual. This does not use the camera at all; it's a
    reflex, not a decision.
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


def proximity_scale(area_frac):
    """1.0 while the target still looks far away, tapering linearly to 0.0
    by the time it's close (area_frac >= STOP_AREA_FRACTION).

    Used to scale down BOTH forward speed and turning together as the car
    approaches. Steering needs full authority while there's room to
    maneuver, but continuing to spin in place right next to the target
    (to chase a slightly off-center centroid) is exactly the kind of motion
    that can catch the target and push it — so turning tapers to zero at
    the same rate as forward speed, and the car just comes to a full,
    non-rotating stop once close, even if not perfectly centered.
    """
    if area_frac >= STOP_AREA_FRACTION:
        return 0.0
    if area_frac <= FORWARD_SLOWDOWN_AREA:
        return 1.0
    return (STOP_AREA_FRACTION - area_frac) / (STOP_AREA_FRACTION - FORWARD_SLOWDOWN_AREA)


def smooth(raw_value, history, ema_prev):
    """Median-filter then exponentially smooth a raw per-frame reading."""
    history.append(raw_value)
    median_value = statistics.median(history)
    return DETECT_EMA_ALPHA * ema_prev + (1 - DETECT_EMA_ALPHA) * median_value


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

cx_history = deque(maxlen=DETECT_MEDIAN_WINDOW)
area_history = deque(maxlen=DETECT_MEDIAN_WINDOW)
ema_cx = 0.0
ema_area = 0.0
last_turn_sign = 1        # which way the target was last seen drifting
search_started_at = None
search_direction = 1

try:
    while cap.isOpened():
        # ── Interrupt: obstacle check takes priority every frame ──
        if cs.reflection() > OBSTACLE_REFLECTION_THRESHOLD:
            avoid_obstacle(dm)
            cx_history.clear(); area_history.clear()  # scene changed, discard stale smoothing
            continue

        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        target = get_target(frame)

        if target is not None:
            search_started_at = None
            ema_cx = smooth(target["cx_norm"], cx_history, ema_cx)
            ema_area = smooth(target["area_frac"], area_history, ema_area)

            scale = proximity_scale(ema_area)
            turn = STEER_GAIN * ema_cx
            turn = max(-STEER_MAX, min(STEER_MAX, turn)) * scale
            forward = FORWARD_MAX_SPEED * scale

            left_speed = forward + turn
            right_speed = forward - turn
            last_turn_sign = 1 if ema_cx > 0 else -1

        else:
            # Target not visible — spin in place to search, favoring the
            # direction it was last seen heading, reversing periodically
            # in case that guess was wrong.
            if search_started_at is None:
                search_started_at = time.time()
                search_direction = last_turn_sign
            elif time.time() - search_started_at > SEARCH_REVERSE_DIRECTION_AFTER:
                search_direction *= -1
                search_started_at = time.time()

            left_speed = SEARCH_TURN_SPEED * search_direction
            right_speed = -SEARCH_TURN_SPEED * search_direction
            # Discard smoothing history while lost so reacquiring the target
            # doesn't start from a stale, no-longer-relevant estimate.
            cx_history.clear()
            area_history.clear()

        dm.movement_move_tank(apply_deadzone(left_speed), apply_deadzone(right_speed))

        debug = draw_debug(frame, target)
        cv2.putText(debug, f"L: {left_speed:+5.1f}  R: {right_speed:+5.1f}",
                    (10, debug.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)
        cv2.imshow("Autonomous Drive", debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.033)  # ~30 Hz control loop

finally:
    dm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
