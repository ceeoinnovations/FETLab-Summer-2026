"""Does the TRAINED POLICY walk straight in simulation?

Runs the exported policy (policy_weights.npz) in the clean sim env - no
domain randomization, observations exactly as a zero-IMU deployment feeds
them - and measures forward distance, lateral drift, and heading (yaw) drift.

    uv run python eval_straightness.py policy_weights3.npz

Verdicts:
  - curves consistently ONE way in sim  -> policy/training problem
  - straight in sim, curves on hardware -> hardware asymmetry or obs pipeline
"""

import sys

import mujoco
import numpy as np

from lego_env_deploy import LegoQuadEnv
from numpy_policy import NumpyPolicy

POLICY = sys.argv[1] if len(sys.argv) > 1 else "policy_weights.npz"
COMMAND = 0.15
EPISODES = 5
EPISODE_S = 30.0

policy = NumpyPolicy(POLICY)
env = LegoQuadEnv(control_dt=0.2, episode_s=EPISODE_S, randomize=False)


def body_yaw(env):
  R = np.zeros(9)
  mujoco.mju_quat2Mat(R, env.data.qpos[3:7])
  R = R.reshape(3, 3)
  return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


print(f"policy: {POLICY} | command {COMMAND} m/s | {EPISODES} x {EPISODE_S:.0f}s episodes")
print(f"{'ep':>3} {'forward(m)':>11} {'lateral(m)':>11} {'yaw drift(deg)':>15} {'fell':>5}")
yaws = []
for ep in range(EPISODES):
  obs, _ = env.reset(seed=200 + ep)
  env.cmd = COMMAND
  x0, y0 = env.data.qpos[0], env.data.qpos[1]
  yaw0 = body_yaw(env)
  fell = False
  for _ in range(env.max_steps):
    # feed EXACTLY what zero-IMU deployment feeds
    obs[0:3] = 0.0                       # lin vel: blind
    obs[3:6] = 0.0                       # gyro: zeros
    obs[6:9] = (0.0, 0.0, -1.0)          # gravity: fixed upright
    action = policy(obs)
    obs, r, term, trunc, info = env.step(action)
    if term:
      fell = True
      break
  dx = env.data.qpos[0] - x0
  dy = env.data.qpos[1] - y0
  dyaw = (body_yaw(env) - yaw0 + 180) % 360 - 180
  yaws.append(dyaw)
  print(f"{ep:>3} {dx:>11.2f} {dy:>11.2f} {dyaw:>15.1f} {str(fell):>5}")

yaws = np.array(yaws)
print(f"\nmean yaw drift {yaws.mean():+.1f} deg/episode "
      f"(std {yaws.std():.1f})")
if abs(yaws.mean()) > 20 and yaws.std() < abs(yaws.mean()):
  print("VERDICT: policy itself curves consistently -> training-side issue")
elif abs(yaws.mean()) < 15:
  print("VERDICT: policy is straight in sim -> hardware-side issue "
        "(motor asymmetry, signs, or obs pipeline)")
else:
  print("VERDICT: inconsistent drift - policy wanders rather than curves; "
        "heading hold on hardware is the right tool")