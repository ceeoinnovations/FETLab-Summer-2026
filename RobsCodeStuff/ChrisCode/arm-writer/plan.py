"""
Step 2 — Plan the arm trajectory for a name.

Loads the trained SAC policy and threads it through the 3-D waypoints
for each letter in config.NAME.  At each waypoint the policy runs until
the LED reaches the target (or a step limit), recording the joint angles
at every step.

Saves trajectory.npz containing:
  joint_angles  — (T, 3) array of joint angles in radians
  led_positions — (T, 3) array of LED XYZ positions in metres
  waypoints     — (W, 3) array of target waypoints (PEN_UP excluded)
  pen_down      — (T,)   boolean: True when LED is drawing

Run visualize.py after this to see the result.
"""

import numpy as np
import mujoco
from stable_baselines3 import SAC
from env import ArmEnv
from letters import name_to_waypoints, PEN_UP
from config import NAME, SUCCESS_THR, MAX_STEPS, CANVAS_Z

POLICY_PATH   = "arm_policy"
LIFT_Z        = CANVAS_Z[1] + 0.01   # pen-up height

print(f"Loading policy from {POLICY_PATH}...")
model = SAC.load(POLICY_PATH)

env = ArmEnv()
obs, _ = env.reset()

# ── Generate waypoints for the name ──────────────────────────────────────────
waypoints = name_to_waypoints(NAME)
print(f"Name: '{NAME}'  →  {len(waypoints)} waypoints "
      f"({sum(1 for w in waypoints if w is not None)} positions, "
      f"{sum(1 for w in waypoints if w is None)} pen-ups)")

# ── Follow waypoints with the policy ─────────────────────────────────────────
all_joints  = []
all_led     = []
all_pen     = []
wp_targets  = []

pen_is_down = True

for wp in waypoints:
    if wp is PEN_UP:
        pen_is_down = False
        continue

    wp_targets.append(wp)
    # Set the environment's target to this waypoint
    env._target    = wp
    pen_is_down_wp = not np.isclose(wp[2], LIFT_Z, atol=0.005)

    obs = env._get_obs()
    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)

        all_joints.append(env.data.qpos[:3].copy())
        all_led.append(env._led_pos())
        all_pen.append(pen_is_down_wp)

        if terminated or info["dist"] < SUCCESS_THR:
            break

    pen_is_down = True   # reset after arrival

env.close()

np.savez(
    "trajectory.npz",
    joint_angles  = np.array(all_joints,  dtype=np.float32),
    led_positions = np.array(all_led,     dtype=np.float32),
    waypoints     = np.array(wp_targets,  dtype=np.float32),
    pen_down      = np.array(all_pen,     dtype=bool),
)
print(f"Trajectory saved → trajectory.npz  ({len(all_joints)} total steps)")
print("Run: python execute.py  to drive the real arm")
print("Run: python visualize.py  to see the result")
