"""
Step 3 — Execute the planned trajectory on the real LEGO arm.

Motor assignment:
  Joint 1 (base yaw)    → singleMotor
  Joint 2 (shoulder)    → doubleMotor, LEFT  motor
  Joint 3 (elbow)       → doubleMotor, RIGHT motor

The joint-angle sequence from plan.py is downsampled to one command per
waypoint (the final joint angles when the sim reached each target).
Each joint is moved by the delta from its current position.

CALIBRATION: measure your physical gear ratios and update
  GEAR_DEG_PER_RAD in config.py before running on the real arm.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import legoeducation as le
from lelib import singleMotor, doubleMotor
from config import SERIAL, GEAR_DEG_PER_RAD, MOTOR_SPEED, WAYPOINT_PAUSE

data    = np.load("trajectory.npz")
joints  = data["joint_angles"]    # (T, 3) radians
pen     = data["pen_down"]         # (T,)   bool

# ── Downsample: take the joint state at the end of each waypoint segment ──────
# Waypoint boundaries = wherever pen_down changes or at the last step
wp_indices = []
for i in range(1, len(pen)):
    if pen[i] != pen[i - 1]:
        wp_indices.append(i - 1)
wp_indices.append(len(joints) - 1)

joint_sequence = joints[wp_indices]           # shape (W, 3)
pen_sequence   = pen[wp_indices]              # shape (W,)

print(f"Trajectory: {len(joint_sequence)} waypoints")

# ── Connect to motors ──────────────────────────────────────────────────────────
sm = singleMotor()
dm = doubleMotor()
print("Connecting to single motor (joint 1)...")
sm.connect(SERIAL)
print("Connecting to double motor (joints 2+3)...")
dm.connect(SERIAL)

# Zero all motor positions
sm.motor_reset_relative_position()
dm.motor_reset_relative_position(motor=le.MOTOR_LEFT)
dm.motor_reset_relative_position(motor=le.MOTOR_RIGHT)
print("Motor positions zeroed.\n")

def rad_to_deg(joint_idx: int, angle_rad: float) -> int:
    key = f"j{joint_idx + 1}"
    return round(angle_rad * GEAR_DEG_PER_RAD[key])

# Track where each motor currently is (in motor degrees from start)
current = [0, 0, 0]

print(f"Writing: '{data.get('name', '?')}' — {len(joint_sequence)} waypoints")
print(f"Speed: {MOTOR_SPEED}%  Pause: {WAYPOINT_PAUSE}s\n")

try:
    for i, (angles, drawing) in enumerate(zip(joint_sequence, pen_sequence)):
        targets = [rad_to_deg(j, angles[j]) for j in range(3)]
        deltas  = [targets[j] - current[j]   for j in range(3)]

        action = "drawing" if drawing else "lifting"
        print(f"  WP {i+1:3d}  {action:8s}  "
              f"J1={targets[0]:+5d}°  J2={targets[1]:+5d}°  J3={targets[2]:+5d}°")

        # Move joint 1 (single motor)
        if abs(deltas[0]) > 2:
            direction = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if deltas[0] > 0 \
                        else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
            sm.motor_run_for_degrees(abs(deltas[0]), direction=direction,
                                     speed=MOTOR_SPEED)

        # Move joints 2 and 3 (double motor, left and right independently)
        if abs(deltas[1]) > 2:
            direction2 = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if deltas[1] > 0 \
                         else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
            dm.motor_run_for_degrees(abs(deltas[1]), direction=direction2,
                                     motor=le.MOTOR_LEFT, speed=MOTOR_SPEED)

        if abs(deltas[2]) > 2:
            direction3 = le.MOTOR_MOVE_DIRECTION_CLOCKWISE if deltas[2] > 0 \
                         else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
            dm.motor_run_for_degrees(abs(deltas[2]), direction=direction3,
                                     motor=le.MOTOR_RIGHT, speed=MOTOR_SPEED,
                                     blocking=True)

        current = targets
        time.sleep(WAYPOINT_PAUSE)

except KeyboardInterrupt:
    print("\nInterrupted.")
finally:
    sm.stop()
    dm.stop()
    print("Done.")
