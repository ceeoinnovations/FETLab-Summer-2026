"""check_reward_floor for the v8 env, handling the 26->28 obs change.

The train6.2 reference policy expects the 26-dim v6 obs; the v8 obs is that
same layout plus 2 clock dims at the end, so we feed the ref policy obs[:26].
PASS iff an upright policy survives the full episode with positive mean and
last-fifth per-step reward (no suicide-collapse incentive).

    uv run python floor_check8.py [imu]
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from lego_env23_1 import LegoQuadEnv
sys.path.insert(0, "..")
from numpy_policy import NumpyPolicy

IMU = sys.argv[1] if len(sys.argv) > 1 else "live"
policy = NumpyPolicy("../train6.2/policy_weights6.npz")
env = LegoQuadEnv(control_dt=0.2, episode_s=30.0, randomize=False)
obs, _ = env.reset(seed=200)
env.cmd = 0.15

rewards = []
for _ in range(env.max_steps):
  obs[0:3] = 0.0
  if IMU == "zero":
    obs[3:6] = 0.0
    obs[6:9] = (0.0, 0.0, -1.0)
  obs, r, term, trunc, info = env.step(policy(obs[:26]))   # strip clock for ref
  rewards.append(r)
  if term:
    break

rewards = np.array(rewards)
n = len(rewards)
tail = rewards[max(0, int(0.8 * n)):]
survived = n >= env.max_steps
ok = survived and rewards.mean() > 0 and tail.mean() > 0
print(f"ref train6.2 in train8 env (imu={IMU}):")
print(f"  survived    {n}/{env.max_steps}  ({'full' if survived else 'FELL early'})")
print(f"  mean reward {rewards.mean():+.3f}/step")
print(f"  last-fifth  {tail.mean():+.3f}/step")
print(f"  VERDICT: {'PASS - floor positive, safe to train' if ok else 'FAIL - collapse risk'}")
sys.exit(0 if ok else 1)
