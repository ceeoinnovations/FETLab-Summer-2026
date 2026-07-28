"""lego_env_deploy17.py
Deployment-matched training environment, v8 (faithful mesh model).

Subclasses lego_env17's LegoQuadEnv (phase-clock trot prior). Adds domain
randomization and the deploy-matched obs. The v8 phase clock is appended to
the obs here too, so the deploy obs matches the base 28-dim layout.

imu_mode: "zero" | "live" | "yaw_random" (see lego_env_deploy6 header).
5 Hz control, blind lin-vel, wide gain/latency/phase randomization.
"""

import numpy as np

from lego_env23_1 import LegoQuadEnv, MAX_CRANK_SPEED  # noqa: F401


class LegoQuadDeployEnv(LegoQuadEnv):
  def __init__(self, imu_mode="zero", **kw):
    kw.setdefault("control_dt", 0.2)  # 5 Hz
    assert imu_mode in ("zero", "live", "yaw_random")
    self.imu_mode = imu_mode
    super().__init__(**kw)

  def reset(self, seed=None, options=None):
    self.speed_scale = np.ones(4)
    self.ang_bias = np.zeros(4)
    self._held = np.zeros(4)
    self._yaw_R = np.eye(3)
    self._w_prev = np.zeros(3)
    obs, info = super().reset(seed=seed, options=options)
    rng = self.np_random
    self.speed_scale = rng.uniform(0.7, 1.3, 4)
    self.ang_bias = rng.uniform(-0.25, 0.25, 4)
    if self.imu_mode == "yaw_random":
      th = rng.uniform(-np.pi, np.pi)
      c, s = np.cos(th), np.sin(th)
      self._yaw_R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return self._obs(), info

  def step(self, action):
    rng = self.np_random
    a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
    if rng.uniform() < 0.25:            # BLE latency: old action sticks
      a = self._held
    self._held = a.copy()
    a = a * self.speed_scale * rng.normal(1.0, 0.05, 4)
    return super().step(np.clip(a, -1, 1))

  def _obs(self):
    rng = self.np_random
    R, v_b, w_b = self._body_frame()
    ang = self.data.qpos[self.crank_qpos] + self.ang_bias
    vel = self.data.qvel[self.crank_qvel] * rng.normal(1.0, 0.05, 4)

    if self.imu_mode == "zero":
      g_obs = np.array([0.0, 0.0, -1.0])
      w_obs = np.zeros(3)
    elif self.imu_mode == "live":
      g_b = R.T @ np.array([0.0, 0.0, -1.0])
      g_obs = g_b + rng.normal(0, 0.03, 3)
      w_smooth = 0.5 * (w_b + self._w_prev)
      self._w_prev = w_b.copy()
      w_obs = w_smooth + rng.normal(0, 0.06, 3)
    else:  # yaw_random
      g_b = R.T @ np.array([0.0, 0.0, -1.0])
      g_obs = self._yaw_R @ g_b + rng.normal(0, 0.05, 3)
      w_obs = self._yaw_R @ w_b + rng.normal(0, 0.08, 3)

    p = 2.0 * np.pi * self._clock_frac()          # v8 phase clock (in obs)
    he = self._heading_err() + rng.normal(0, 0.03)   # v15: heading err (+IMU noise)
    return np.concatenate([
      np.zeros(3),
      w_obs, g_obs,
      np.sin(ang), np.cos(ang),
      vel / 10.0,
      self.prev_action, [self.cmd], [np.sin(p), np.cos(p)],
      [np.sin(he), np.cos(he)],
    ])
