"""Deployment-matched training environment.

Differences from LegoQuadEnv, chosen to match the real LEGO hardware:
  - base linear velocity is ZEROED in the observation (no sensor can
    measure it on the robot); the reward still uses the true value.
  - 20 Hz control (BLE-realistic) instead of 50 Hz.
  - extra domain randomization: per-episode motor speed-scale error and
    IMU gyro noise, so the policy tolerates imperfect calibration.
"""

import numpy as np

from lego_env import LegoQuadEnv, MAX_CRANK_SPEED  # noqa: F401


class LegoQuadDeployEnv(LegoQuadEnv):
  def __init__(self, **kw):
    kw.setdefault("control_dt", 0.05)  # 20 Hz, BLE-realistic
    super().__init__(**kw)

  def reset(self, seed=None, options=None):
    obs, info = super().reset(seed=seed, options=options)
    # per-episode actuator gain error (calibration is never perfect)
    self.speed_scale = self.np_random.uniform(0.8, 1.2, 4)
    return self._obs(), info

  def step(self, action):
    scaled = np.clip(np.asarray(action, dtype=np.float64), -1, 1) * self.speed_scale
    return super().step(np.clip(scaled, -1, 1))

  def _obs(self):
    obs = super()._obs()
    obs[0:3] = 0.0                                   # blind to base lin vel
    obs[3:6] += self.np_random.normal(0, 0.05, 3)    # gyro noise
    return obs
