"""Deployment-matched training environment, v4.

imu_mode:
  "zero"       - gravity fixed (0,0,-1), gyro zeros; matches IMU_SOURCE="zero".
  "live"       - real gravity + gyro with hardware texture: noise plus
                 one-step smoothing that mimics the fused-angle finite
                 differencing at 5 Hz. Matches IMU_SOURCE="fused" deployment
                 (front unit, yaw face set, verified pitch/roll/yaw semantics).
  "yaw_random" - random per-episode rotation about vertical on gravity/gyro;
                 for when IMU axes are unknown (superseded by verified fused
                 semantics, kept for reference).

Also: 5 Hz control, blind lin-vel, wide gain/latency/phase randomization.
"""

import numpy as np

from lego_env5 import LegoQuadEnv, MAX_CRANK_SPEED  # noqa: F401


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
      # finite-difference texture: rates are the average over the last step,
      # not instantaneous -> one-step smoothing, plus fused-angle jitter
      w_smooth = 0.5 * (w_b + self._w_prev)
      self._w_prev = w_b.copy()
      w_obs = w_smooth + rng.normal(0, 0.06, 3)
    else:  # yaw_random
      g_b = R.T @ np.array([0.0, 0.0, -1.0])
      g_obs = self._yaw_R @ g_b + rng.normal(0, 0.05, 3)
      w_obs = self._yaw_R @ w_b + rng.normal(0, 0.08, 3)

    return np.concatenate([
      np.zeros(3),
      w_obs, g_obs,
      np.sin(ang), np.cos(ang),
      vel / 10.0,
      self.prev_action, [self.cmd],
    ])