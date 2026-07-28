"""
Gymnasium environment wrapping the MuJoCo 3-DOF arm.

Observation (12-D):
  [j1, j2, j3,              — current joint angles (rad)
   j1_dot, j2_dot, j3_dot,  — joint velocities (rad/s)
   tx, ty, tz,              — target LED position (m)
   ex, ey, ez]              — error vector: LED - target (m)

Action (3-D, each in [-1, 1]):
  Normalised target angles for the three position servos.
  Scaled to the actuator ctrlrange before being passed to MuJoCo.

Reward:
  -distance        per step  (dense guidance)
  +5.0             on success (LED within SUCCESS_THR)
  -0.002 * |qvel|² per step  (velocity regularisation for smooth motion)
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from config import (ARM_XML, MAX_STEPS, SUBSTEPS, SUCCESS_THR,
                    CANVAS_X, CANVAS_Y, CANVAS_Z)


class ArmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model       = mujoco.MjModel.from_xml_path(ARM_XML)
        self.data        = mujoco.MjData(self.model)
        self.render_mode = render_mode

        self._led_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "led"
        )
        self._n_act  = self.model.nu          # 3
        self._lo     = self.model.actuator_ctrlrange[:, 0]
        self._hi     = self.model.actuator_ctrlrange[:, 1]

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(self._n_act,), dtype=np.float32
        )

        self._target  = np.zeros(3)
        self._step_n  = 0

        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self._viewer = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _led_pos(self):
        return self.data.site_xpos[self._led_id].copy()

    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Map [-1, 1] to actuator control range."""
        return self._lo + (action + 1.0) * 0.5 * (self._hi - self._lo)

    def _get_obs(self) -> np.ndarray:
        err = self._led_pos() - self._target
        return np.concatenate([
            self.data.qpos[:self._n_act],
            self.data.qvel[:self._n_act],
            self._target,
            err,
        ]).astype(np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self._target = np.array([
            CANVAS_X,
            self.np_random.uniform(*CANVAS_Y),
            self.np_random.uniform(*CANVAS_Z),
        ])
        self._step_n = 0

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        self.data.ctrl[:] = self._scale_action(np.clip(action, -1, 1))

        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)

        led    = self._led_pos()
        dist   = float(np.linalg.norm(led - self._target))
        vel_sq = float(np.sum(np.square(self.data.qvel[:self._n_act])))

        success = dist < SUCCESS_THR
        reward  = -dist - 0.002 * vel_sq + (5.0 if success else 0.0)

        self._step_n += 1
        terminated   = success
        truncated    = self._step_n >= MAX_STEPS

        if self._viewer is not None:
            self._viewer.sync()

        return self._get_obs(), reward, terminated, truncated, {
            "dist": dist, "success": success
        }

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
