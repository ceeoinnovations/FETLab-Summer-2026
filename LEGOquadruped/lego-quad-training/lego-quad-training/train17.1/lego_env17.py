"""Gymnasium environment for the LEGO quadruped - FAITHFUL MESH MODEL, v9.

v9 = concept (1) + concept (3) of paper 2409.15780, stacked on train8's clock:

  (1) PHASE-CLOCK trot prior (from v8, unchanged): a clock advancing at period
      T=0.72 s is in the OBSERVATION (sin/cos), and a contact-barrier gait term
      rewards on-clock foot contact (f_i = +g_i in stance, -g_i in swing;
      g_i = sin(2*pi*(phase+phi_i)), diagonal phi={FL:0,BR:0,FR:.5,BL:.5};
      barrier keeps f_i >= -0.6). Reset is phase-aligned to the assembled rest
      pose (no random crank splay).

  (3) MULTIPLICATIVE reward (NEW): reward = r_pos * exp(0.2 * r_neg), where
      r_pos = velocity tracking (2*track + progress, >= 0) and r_neg = all
      penalties (lateral, upright, smooth, balance, heading, gait barrier;
      <= 0). This removes the standing subsidy that made train8 stand still and
      ignore the clock: standing => r_pos ~ 0 => reward ~ 0, so forward motion
      is the ONLY way to earn. It also removes the additive `alive` bonus
      entirely - because reward >= 0 on every surviving step, falling-to-exit
      is never optimal (the -5 terminal penalty is strictly worse than any
      survivable step), so no suicide-collapse even without concept (2).

Concept (2), the two-critic split that lets the gait barrier be a sharp
separate signal, is still deferred; here the barrier rides inside r_neg (so it
nudges via exp rather than dominating). Single change vs train8: the reward
structure.
"""

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

MAX_CRANK_SPEED = 12.0  # rad/s
XML_PATH = Path(__file__).parent / "lego_quad_mesh.xml"
GEAR_JOINTS = ("frontLeftGear", "frontRightGear", "backLeftGear", "backRightGear")

# --- phase-clock trot prior (paper 2409.15780) ---
CLOCK_T = 0.72                  # gait period (s)
# diagonal-trot phase offset per gear [FL, FR, BL, BR]; calibrated on the mesh:
# {FL,BR} share phase, {FR,BL} half a cycle apart.
GAIT_PHI = np.array([0.0, 0.5, 0.5, 0.0])
GAIT_GATE = -0.6                # barrier lower bound d_gait_lower
GAIT_WEIGHT = 2.5               # barrier weight (routed to the 2nd critic)
# v12: linear forward-velocity reward weight. At the (clipped) commanded speed
# ~0.04 m/s this gives r_pos ~2.0; it is 0 at standing, so R_standard=0 for a
# non-translating policy - killing the "trot in place" optimum train11 found.
FWD_WEIGHT = 50.0
# v13: walk the OTHER way. Forward = -v_b[0] (train12 walked +v_b[0]); this is
# the robot's original intended direction. The reward and the RSI crank drive
# are both flipped so they agree on this direction.
FWD_SIGN = -1.0
# v15: the heading error is now IN THE OBS (sin/cos of yaw - yaw0), so the policy
# can SEE drift and learn to steer back - train12-14 were heading-blind (gyro
# gives yaw rate but nothing tells them which way is straight), so they couldn't
# self-correct. With drift observable, a moderate heading penalty now actually
# drives closed-loop steering. Deploy feeds the IMU (yaw - yaw0) into these dims.
HEADING_WEIGHT = 2.0    # moderate - now actionable (was 1.0, unobservable)

# --- v11 reference-state initialization (RSI) ---
# The discovery wall: PPO never finds the coordinated trot from a standing start
# (train8-10 all plateaued at standing). RSI starts most episodes already
# trotting and translating - the open-loop phased crank drive that diag_contact
# showed moves the body forward - so the policy REFINES a gait instead of
# discovering one cold. A minority of episodes still cold-start (assembled rest)
# so the policy also learns to initiate the gait from standstill (deployment).
RSI_PROB = 0.85                 # fraction of episodes seeded mid-trot
RSI_WARMUP_CYCLES = 2.0         # crank revolutions to reach steady gait pre-policy
# per-gear drive dir [FL,FR,BL,BR]. [+,-,+,-] is the crank direction whose gait
# translates in +v_b[0] - the reward's forward. (The other direction travels in
# -v_b[0], which the reward would penalize; verified both directions move.)
RSI_SIGNS = np.array([-1.0, 1.0, -1.0, 1.0])   # v13: reversed - seeds -v_b[0]
# v17: FRONT-BACK balance. train15 was back-wheel-drive (front jittered, back
# drove); train16's per-leg rotation floor forced all legs fast but killed
# steering (steering needs to SLOW one side). The right constraint is to make
# the FRONT legs drive as much as the BACK (use all four), while leaving
# LEFT-RIGHT free (slowing a side to steer keeps front-back matched). So we
# penalize only front driving LESS than back - not absolute speed, not L/R.
FB_WEIGHT = 0.4
# leg body carrying each gear's foot, in GEAR_JOINTS order (for contact sensing)
LEG_BODIES = ("lleg", "rleg_2", "lleg_2", "rleg")   # FL, FR, BL, BR


class LegoQuadEnv(gym.Env):
  metadata = {"render_modes": []}

  def __init__(
    self,
    cmd_range=(0.01, 0.04),   # v11: realistic - the linkage tops out ~0.02 m/s
                              # (matches the real robot); 0.05-0.35 was unreachable
    episode_s: float = 10.0,
    control_dt: float = 0.02,
    randomize: bool = True,
  ):
    self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    self.data = mujoco.MjData(self.model)
    self.cmd_range = cmd_range
    self.randomize = randomize
    self.n_substeps = int(round(control_dt / self.model.opt.timestep))
    self.control_dt = control_dt
    self.max_steps = int(episode_s / control_dt)

    self.crank_qpos = np.array(
      [self.model.jnt_qposadr[self.model.joint(n).id] for n in GEAR_JOINTS]
    )
    self.crank_qvel = np.array(
      [self.model.jnt_dofadr[self.model.joint(n).id] for n in GEAR_JOINTS]
    )
    self._floor_geom = self.model.geom("floor").id

    # collision geoms grouped per leg (for foot-contact detection)
    body_leg = {b: i for i, b in enumerate(LEG_BODIES)}
    self._leg_geoms = [[] for _ in LEG_BODIES]
    for gid in range(self.model.ngeom):
      b = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                            self.model.geom_bodyid[gid])
      if b in body_leg and self.model.geom_contype[gid]:
        self._leg_geoms[body_leg[b]].append(gid)

    # v17: the rubber FOOT geoms (grippy). Domain-randomize THEIR friction
    # (not the floor's) so the policy is robust to the real rubber-surface grip.
    self._foot_geoms = [gid for gid in range(self.model.ngeom)
                        if self.model.geom_dataid[gid] >= 0
                        and "foot" in (mujoco.mj_id2name(
                            self.model, mujoco.mjtObj.mjOBJ_MESH,
                            self.model.geom_dataid[gid]) or "")
                        and self.model.geom_contype[gid]]

    self.observation_space = gym.spaces.Box(-np.inf, np.inf, (30,), np.float64)
    self.action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

  # ----- helpers -----

  def _body_frame(self):
    quat = self.data.qpos[3:7]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    return R, R.T @ self.data.qvel[0:3], R.T @ self.data.qvel[3:6]

  def _clock_frac(self):
    """Global gait-clock phase in [0,1)."""
    return (self.phase_t / CLOCK_T) % 1.0

  def _heading_err(self):
    """Signed yaw error from the target heading (v15: makes drift observable)."""
    return self._wrap_pi(self._yaw() - self._yaw0)

  def _obs(self):
    R, v_b, w_b = self._body_frame()
    g_b = R.T @ np.array([0.0, 0.0, -1.0])
    ang = self.data.qpos[self.crank_qpos]
    vel = self.data.qvel[self.crank_qvel]
    p = 2.0 * np.pi * self._clock_frac()
    he = self._heading_err()      # v15: heading error in obs -> policy can steer
    return np.concatenate(
      [v_b, w_b, g_b, np.sin(ang), np.cos(ang), vel / 10.0,
       self.prev_action, [self.cmd], [np.sin(p), np.cos(p)],
       [np.sin(he), np.cos(he)]]
    )

  def _foot_contacts(self):
    """Per-leg contact flag (any leg collision geom touching the floor)."""
    hit = np.zeros(4)
    for c in self.data.contact[: self.data.ncon]:
      other = (c.geom1 if c.geom2 == self._floor_geom
               else c.geom2 if c.geom1 == self._floor_geom else -1)
      if other < 0:
        continue
      for i, gs in enumerate(self._leg_geoms):
        if other in gs:
          hit[i] = 1.0
    return hit

  def _gait_barrier(self):
    """Barrier reward (<=0) enforcing on-clock foot contact (paper concept 1)."""
    frac = self._clock_frac()
    g = np.sin(2.0 * np.pi * (frac + GAIT_PHI))     # gait function per leg
    c = self._foot_contacts()
    f = np.where(c > 0.5, g, -g)                    # +g in stance, -g in swing
    return float(np.mean(np.minimum(0.0, f - GAIT_GATE)))

  def _tilt(self):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
    return float(np.arccos(np.clip(R.reshape(3, 3)[2, 2], -1.0, 1.0)))

  def _yaw(self):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, self.data.qpos[3:7])
    R = R.reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))

  @staticmethod
  def _wrap_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

  # ----- gym API -----

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    mujoco.mj_resetData(self.model, self.data)   # assembled pose (welds intact)
    self.cmd = float(self.np_random.uniform(*self.cmd_range))
    self.prev_action = np.zeros(4)
    self.steps = 0
    self.phase_t = 0.0                            # phase-aligned clock start

    if self.randomize:
      # v17: randomize the RUBBER FOOT friction around its grippy baseline (2.0)
      # so the policy is robust to the real rubber-surface grip. NO random crank
      # angles - the assembled rest pose is the phase-aligned reference; random
      # crank qpos breaks the weld linkage and destroys coordination (v6/v7 bug).
      mu = self.np_random.uniform(0.6, 1.8)   # v17.1: surface variety (slippery->grippy) + foot uncertainty
      for gid in self._foot_geoms:
        self.model.geom_friction[gid, 0] = mu

    for _ in range(400):                          # settle onto the feet
      mujoco.mj_step(self.model, self.data)

    # v11 reference-state init: pre-roll into a moving trot on most episodes so
    # the policy starts in (not has to discover) forward locomotion.
    if self.np_random.uniform() < RSI_PROB:
      omega = 2.0 * np.pi / CLOCK_T                # crank speed = one cycle per T
      phi0 = float(self.np_random.uniform(0.0, 1.0))
      n_pre = int(round((RSI_WARMUP_CYCLES + phi0) * CLOCK_T / self.model.opt.timestep))
      for _ in range(n_pre):
        self.data.ctrl[:] = RSI_SIGNS * omega
        mujoco.mj_step(self.model, self.data)
      self.phase_t = phi0 * CLOCK_T                # align clock to the pre-rolled phase
      self.data.ctrl[:] = 0.0

    self._yaw0 = self._yaw()
    return self._obs(), {}

  def step(self, action):
    action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    self.data.ctrl[:] = action * MAX_CRANK_SPEED
    for _ in range(self.n_substeps):
      mujoco.mj_step(self.model, self.data)
    self.steps += 1
    self.phase_t += self.control_dt               # advance the gait clock

    _, v_b, w_b = self._body_frame()
    tilt = self._tilt()

    # --- v12: TRANSLATION-DOMINANT task reward (+ two-critic barrier) ---
    # r_pos is a pure LINEAR forward-velocity reward, ZERO at standing. train11's
    # track-Gaussian rewarded v~cmd, but at low commands v=0 scored high too, so
    # the policy trotted ON-CLOCK IN PLACE (clock-match 69% but net ~0 travel).
    # A linear forward reward can only be earned by actually carrying the body
    # forward, so "trot that translates" strictly beats "trot in place".
    fwd = FWD_SIGN * v_b[0]                        # v13: forward is -v_b[0]
    r_pos = FWD_WEIGHT * float(np.clip(fwd, 0.0, self.cmd))   # 0 standing, ~2 at cmd
    lateral = -0.3 * abs(v_b[1]) - 0.2 * abs(w_b[2])
    upright = -0.5 * tilt
    smooth = -0.02 * float(np.sum((action - self.prev_action) ** 2))
    # v17: front-back balance - penalize the front legs net-driving less than
    # the back (in trot direction), so all four contribute; L/R stays free.
    net_rot = RSI_SIGNS * self.data.qvel[self.crank_qvel]
    front_drive = 0.5 * (net_rot[0] + net_rot[1])   # FL, FR
    back_drive = 0.5 * (net_rot[2] + net_rot[3])     # BL, BR
    balance = -FB_WEIGHT * max(0.0, float(back_drive - front_drive))
    heading = -HEADING_WEIGHT * abs(self._heading_err())
    r_neg = lateral + upright + smooth + balance + heading   # gait NOT here
    r_standard = float(r_pos * np.exp(0.2 * r_neg))
    if terminated_flag := (tilt > np.radians(70.0)
                           or not np.isfinite(self.data.qpos).all()):
      r_standard -= 5.0
    # R_barrier = phase-clock contact barrier, routed to the SECOND critic via
    # info["r_barrier"] (concept 2). Kept out of R_standard so its sharp penalty
    # gets its own value fn + separately-normalized advantage and can shape the
    # gait hard without making falling-to-exit look good. Weight is ~arbitrary:
    # the two-critic normalizes each advantage stream, so scale washes out.
    r_barrier = float(GAIT_WEIGHT * self._gait_barrier())

    self.prev_action = action.copy()
    terminated = bool(terminated_flag)
    truncated = self.steps >= self.max_steps

    return self._obs(), r_standard, terminated, truncated, {
      "forward_vel": float(FWD_SIGN * v_b[0]), "command": self.cmd,
      "r_barrier": r_barrier, "gait": r_barrier,
    }
