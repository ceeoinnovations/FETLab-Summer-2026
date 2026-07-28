"""Deployment-matched training environment, v2.

Matched to measured hardware reality:
  - 5 Hz control (BLE round-trips measured ~300 ms at 20 Hz attempt)
  - base linear velocity blinded (unmeasurable)
  - WIDE randomization: motor gain error (per-episode AND per-step),
    random action latency (BLE jitter), gyro noise, and a per-episode
    crank-angle bias (imperfect phase anchoring on real hardware).
"""

import numpy as np

from lego_env import LegoQuadEnv, MAX_CRANK_SPEED  # noqa: F401


class LegoQuadDeployEnv(LegoQuadEnv):
  def __init__(self, **kw):
    kw.setdefault("control_dt", 0.2)  # 5 Hz, BLE-realistic
    super().__init__(**kw)

  def reset(self, seed=None, options=None):
    # parent reset calls _obs(), so randomization fields must exist first
    self.speed_scale = np.ones(4)
    self.ang_bias = np.zeros(4)
    self._held = np.zeros(4)
    obs, info = super().reset(seed=seed, options=options)
    rng = self.np_random
    self.speed_scale = rng.uniform(0.7, 1.3, 4)      # calibration error
    self.ang_bias = rng.uniform(-0.25, 0.25, 4)      # phase-anchor error (rad)
    return self._obs(), info

  def step(self, action):
    rng = self.np_random
    a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
    if rng.uniform() < 0.25:                         # BLE latency: old action sticks
      a = self._held
    self._held = a.copy()
    a = a * self.speed_scale * rng.normal(1.0, 0.05, 4)   # gain error + jitter
    return super().step(np.clip(a, -1, 1))

  def _obs(self):
    # rebuild obs with hardware-like corruptions
    rng = self.np_random
    R, v_b, w_b = self._body_frame()
    g_b = R.T @ np.array([0.0, 0.0, -1.0])
    ang = self.data.qpos[self.crank_qpos] + self.ang_bias
    vel = self.data.qvel[self.crank_qvel] * rng.normal(1.0, 0.05, 4)
    return np.concatenate([
      np.zeros(3),                                   # blind lin vel
      w_b + rng.normal(0, 0.08, 3),                  # gyro noise
      g_b + rng.normal(0, 0.03, 3),
      np.sin(ang), np.cos(ang),
      vel / 10.0,
      self.prev_action, [self.cmd],
    ])
