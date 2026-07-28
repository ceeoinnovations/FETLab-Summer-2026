"""Gymnasium environment for the LEGO crank-walker quadruped (CPU MuJoCo).

Observation (26,):
  base linear velocity (body frame)   3
  base angular velocity (body frame)  3
  projected gravity (body frame)      3
  sin(crank angles)                   4   (angles are unbounded; sin/cos keeps obs bounded)
  cos(crank angles)                   4
  crank velocities / 10               4
  previous action                     4
  commanded forward speed             1

Action (4,): crank angular-velocity commands in [-1, 1], scaled to
+/- MAX_CRANK_SPEED rad/s. One action per real motor channel.

Reward: track the commanded forward speed, stay upright, don't jerk.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

MAX_CRANK_SPEED = 12.0  # rad/s
XML_PATH = Path(__file__).parent / "lego_quad_cpu.xml"


class LegoQuadEnv(gym.Env):
  metadata = {"render_modes": []}

  def __init__(
    self,
    cmd_range=(0.05, 0.35),  # forward speed command sampled per episode (m/s)
    episode_s: float = 10.0,
    control_dt: float = 0.02,  # 50 Hz policy
    randomize: bool = True,
  ):
    self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    self.data = mujoco.MjData(self.model)
    self.cmd_range = cmd_range
    self.randomize = randomize
    self.n_substeps = int(round(control_dt / self.model.opt.timestep))
    self.max_steps = int(episode_s / control_dt)

    self.crank_qpos = np.array(
      [self.model.jnt_qposadr[self.model.joint(n).id]
       for n in ("FL_crank", "FR_crank", "HL_crank", "HR_crank")]
    )
    self.crank_qvel = np.array(
      [self.model.jnt_dofadr[self.model.joint(n).id]
       for n in ("FL_crank", "FR_crank", "HL_crank", "HR_crank")]
    )
    self._floor_geom = self.model.geom("floor").id
    self._default_friction = self.model.geom_friction[self._floor_geom].copy()

    self.observation_space = gym.spaces.Box(-np.inf, np.inf, (26,), np.float64)
    self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

  # ----- helpers -----

  def _body_frame(self):
    """Rotation matrix world->body and body-frame velocities."""
    quat = self.data.qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    v_world = self.data.qvel[0:3]
    w_world = self.data.qvel[3:6]
    return R, R.T @ v_world, R.T @ w_world

  def _obs(self):
    R, v_b, w_b = self._body_frame()
    g_b = R.T @ np.array([0.0, 0.0, -1.0])
    ang = self.data.qpos[self.crank_qpos]
    vel = self.data.qvel[self.crank_qvel]
    return np.concatenate(
      [v_b, w_b, g_b, np.sin(ang), np.cos(ang), vel / 10.0,
       self.prev_action, [self.cmd]]
    )

  def _tilt(self):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
    return float(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0)))

  # ----- gym API -----

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    mujoco.mj_resetData(self.model, self.data)
    self.cmd = float(self.np_random.uniform(*self.cmd_range))
    self.prev_action = np.zeros(4)
    self.steps = 0

    if self.randomize:
      # light domain randomization: floor friction and small start jitter
      fr = self._default_friction.copy()
      fr[0] = self.np_random.uniform(0.7, 1.8)
      self.model.geom_friction[self._floor_geom] = fr
      self.data.qpos[self.crank_qpos] = self.np_random.uniform(-np.pi, np.pi, 4)

    # settle onto the feet
    for _ in range(100):
      mujoco.mj_step(self.model, self.data)
    return self._obs(), {}

  def step(self, action):
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    self.data.ctrl[:] = action * MAX_CRANK_SPEED
    for _ in range(self.n_substeps):
      mujoco.mj_step(self.model, self.data)
    self.steps += 1

    _, v_b, w_b = self._body_frame()
    tilt = self._tilt()

    # --- reward ---
    track = np.exp(-((v_b[0] - self.cmd) / 0.15) ** 2)      # follow the command
    lateral = -0.5 * abs(v_b[1]) - 0.05 * abs(w_b[2])        # go straight
    upright = -0.5 * tilt                                     # stay level
    smooth = -0.05 * float(np.sum((action - self.prev_action) ** 2))
    reward = 2.0 * track + lateral + upright + smooth

    self.prev_action = action.copy()

    fell = tilt > np.radians(70.0)
    bad = not np.isfinite(self.data.qpos).all()
    terminated = bool(fell or bad)
    truncated = self.steps >= self.max_steps
    if terminated:
      reward -= 5.0

    return self._obs(), float(reward), terminated, truncated, {
      "forward_vel": float(v_b[0]), "command": self.cmd,
    }
