"""Print what the policy commands EACH leg, per episode.

    uv run python action_stats.py policy_weights3.npz live

If a leg were 'not included', its row would show ~zero mean AND ~zero std.
Healthy: all four legs with similar magnitude and spread.
"""
import sys
import numpy as np
from lego_env import LegoQuadEnv
from numpy_policy import NumpyPolicy

POLICY = sys.argv[1] if len(sys.argv) > 1 else "policy_weights.npz"
IMU = sys.argv[2] if len(sys.argv) > 2 else "live"
policy = NumpyPolicy(POLICY)
env = LegoQuadEnv(control_dt=0.2, episode_s=30.0, randomize=False)
obs, _ = env.reset(seed=42)
env.cmd = 0.15
acts = []
for _ in range(env.max_steps):
  obs[0:3] = 0.0
  if IMU == "zero":
    obs[3:6] = 0.0; obs[6:9] = (0.0, 0.0, -1.0)
  a = policy(obs)
  acts.append(np.clip(a, -1, 1))
  obs, r, term, trunc, info = env.step(acts[-1])
  if term: break
acts = np.array(acts)
print(f"{POLICY} ({IMU}) - action stats over {len(acts)} steps "
      f"(x12 = crank rad/s):")
for i, leg in enumerate(("FL", "FR", "HL", "HR")):
  a = acts[:, i]
  print(f"  {leg}: mean {a.mean():+.2f}  std {a.std():.2f}  "
        f"range [{a.min():+.2f}, {a.max():+.2f}]")