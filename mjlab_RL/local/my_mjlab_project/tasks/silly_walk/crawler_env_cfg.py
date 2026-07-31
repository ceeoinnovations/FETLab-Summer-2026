"""
Lego "crawler" task: TechElementSillyWalk (2-DOF flapping-arm robot,
DoubleMotor hardware) learns to drag its own body forward across the
ground, using an action/observation space designed to transfer to the
real motor.run_left(speed)/run_right(speed) velocity-command interface
and the DoubleMotor's onboard sensors (per-motor position/speed, fused
IMU yaw/pitch/roll).
"""

import math
from pathlib import Path
from dataclasses import dataclass

import mujoco
import torch

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointVelocityActionCfg
from mjlab.envs.mdp.actions.actions import JointVelocityAction
from mjlab.envs.mdp.observations import (
    joint_pos_rel,
    joint_vel_rel,
    projected_gravity,
    last_action,
    base_lin_vel,
)
from mjlab.envs.mdp.terminations import time_out
from mjlab.envs.mdp import rewards as mdp_rewards
from mjlab.managers import (
    ObservationTermCfg,
    ObservationGroupCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
    SceneEntityCfg,
)
from mjlab.scene import SceneCfg
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg

# UNVERIFIED -- see module docstring.
from mjlab.envs.mdp.events import reset_joints_by_offset, reset_root_state_uniform

_ROBOT_XML = Path(__file__).parent / "robot.xml"
_JOINT_NAMES = ("rightAngle", "leftAngle")

# ---------------------------------------------------------------------------
# Real hardware spec: motor.run_left/right(speed), speed in [-100, 100],
# 100 ~= 115 RPM. Deadzone: commands below |10| don't move the real motor
# (static friction). Control loop: 10 Hz.
# ---------------------------------------------------------------------------
_LEGO_MOTOR_MAX_RPM = 115.0
_LEGO_MOTOR_MAX_VEL_RAD_S = _LEGO_MOTOR_MAX_RPM * 2 * math.pi / 60.0  # ~12.0428 rad/s
_DEADZONE_COMMAND_FRACTION = 0.10
_CONTROL_HZ = 10.0
_CONTROL_STEP_S = 1.0 / _CONTROL_HZ  # 0.1s


def _get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(_ROBOT_XML))
    spec.compiler.meshdir = str(_ROBOT_XML.parent / "assets")
    return spec


_ROBOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=tuple(_JOINT_NAMES)),),
)

# Placeholder -- the exact collision-geometry bounding box wasn't easy to
# derive from the mesh data alone. Tune by watching `uv run play`: raise
# if it spawns clipped into the floor, lower if it spawns visibly floating.
_SPAWN_HEIGHT = 0.05

_ROBOT_INIT = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, _SPAWN_HEIGHT),
    rot=(1.0, 0.0, 0.0, 0.0),  # identity -- tune if the CAD "up" doesn't
                               # match world z at this orientation
    lin_vel=(0.0, 0.0, 0.0),
    ang_vel=(0.0, 0.0, 0.0),
    joint_pos={name: 0.0 for name in _JOINT_NAMES},
    joint_vel={name: 0.0 for name in _JOINT_NAMES},
)


def _get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_ROBOT_ARTICULATION,
        init_state=_ROBOT_INIT,
    )


arm_cfg = SceneEntityCfg("crawler", joint_names=_JOINT_NAMES)
root_cfg = SceneEntityCfg("crawler")  # root-state accessors read directly
                                      # off the entity, no body_names/
                                      # body_ids resolution needed


# ---------------------------------------------------------------------------
# World-frame root state -- no built-in mjlab observation function exposes
# world-frame velocity (by design: it's sim-only/privileged, which is
# exactly why it's used below only in reward functions and the critic
# observation group, never the actor). CONFIRMED field names, see docstring.
# ---------------------------------------------------------------------------
def _root_quat_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_quat_w


def _root_lin_vel_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_lin_vel_w


def imu_yaw(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    [B, 1] yaw (radians) extracted from the ground-truth root quaternion.
    Paired with the confirmed built-in `projected_gravity` (pitch/roll-
    equivalent tilt) to approximate the DoubleMotor's fused yaw/pitch/roll
    reading. See module docstring for the yaw-specific sim2real caveat.
    """
    w, x, y, z = _root_quat_w(env, asset_cfg).unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return yaw.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Actions -- velocity control with a deadzone matching real motor behavior.
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class DeadzoneJointVelocityActionCfg(JointVelocityActionCfg):
    """
    Same as JointVelocityActionCfg, but zeroes any per-joint velocity
    command whose magnitude (as a fraction of `scale`) falls below the
    real motor's static-friction deadband. Training against an ideal,
    deadzone-free velocity actuator would teach the policy to rely on
    fine control authority the real motor doesn't have.
    """

    deadzone_fraction: float = _DEADZONE_COMMAND_FRACTION

    def build(self, env) -> "DeadzoneJointVelocityAction":
        return DeadzoneJointVelocityAction(self, env)


class DeadzoneJointVelocityAction(JointVelocityAction):
    cfg: DeadzoneJointVelocityActionCfg

    def apply_actions(self) -> None:
        command_fraction = self._processed_actions / self.cfg.scale
        below_deadzone = torch.abs(command_fraction) < self.cfg.deadzone_fraction
        velocity_target = torch.where(
            below_deadzone, torch.zeros_like(self._processed_actions), self._processed_actions
        )
        self._entity.set_joint_velocity_target(velocity_target, joint_ids=self._target_ids)


actions = {
    "arm_joints": DeadzoneJointVelocityActionCfg(
        entity_name="crawler",
        actuator_names=_JOINT_NAMES,
        scale=_LEGO_MOTOR_MAX_VEL_RAD_S,  # action=1.0 -> full +100 command -> ~12.04 rad/s
        offset=0.0,
        use_default_offset=False,  # keep offset exactly 0, not default_joint_vel
        clip={name: (-_LEGO_MOTOR_MAX_VEL_RAD_S, _LEGO_MOTOR_MAX_VEL_RAD_S) for name in _JOINT_NAMES},  # hard safety net
        deadzone_fraction=_DEADZONE_COMMAND_FRACTION,
    ),
}

# ---------------------------------------------------------------------------
# Observations -- actor sees only what the real DoubleMotor + IMU can
# provide; critic additionally sees privileged sim-only ground truth
# (true world/body-frame velocity -- no GPS/mocap exists on hardware).
# ---------------------------------------------------------------------------
actor_terms = {
    "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": arm_cfg}),
    "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": arm_cfg}),
    "projected_gravity": ObservationTermCfg(func=projected_gravity, params={"asset_cfg": root_cfg}),
    "imu_yaw": ObservationTermCfg(func=imu_yaw, params={"asset_cfg": root_cfg}),
    "last_action": ObservationTermCfg(func=last_action, params={"action_name": None}),
}

critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(func=base_lin_vel, params={"asset_cfg": root_cfg}),
}

observations = {
    "actor": ObservationGroupCfg(actor_terms),
    "critic": ObservationGroupCfg(critic_terms),
}

# ---------------------------------------------------------------------------
# Rewards -- maximize raw forward (world +x) speed, no target to track.
# ---------------------------------------------------------------------------
_MAX_PLAUSIBLE_SPEED_MS = 1.5  # generous safety-net bound, not a real
                                # measurement -- guards against a physics
                                # glitch producing exploitable free reward


def forward_progress_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Raw world-frame +x velocity. No target to track -- just maximize progress."""
    return _root_lin_vel_w(env, asset_cfg)[:, 0]


def excess_speed_penalty(
    env, asset_cfg: SceneEntityCfg, max_speed: float = _MAX_PLAUSIBLE_SPEED_MS
) -> torch.Tensor:
    """Zero cost below max_speed, quadratic above -- a safety net, not a real limit."""
    forward_vel = _root_lin_vel_w(env, asset_cfg)[:, 0]
    excess = torch.clamp(torch.abs(forward_vel) - max_speed, min=0.0)
    return torch.square(excess)


def upright_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """xy-magnitude of projected gravity in body frame -- 0 upright, grows as it tips."""
    grav_b = projected_gravity(env, asset_cfg)
    return torch.sum(torch.square(grav_b[:, :2]), dim=-1)


def lateral_drift_penalty(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Discourages sideways (world y) drift without competing with forward reward."""
    lateral_vel = _root_lin_vel_w(env, asset_cfg)[:, 1]
    return torch.square(lateral_vel)


def idle_penalty(env, asset_cfg: SceneEntityCfg, min_speed: float = 0.03) -> torch.Tensor:
    """
    Positive cost whenever forward speed stays below `min_speed`, zero
    once above it -- mirror image of excess_speed_penalty. Makes standing
    still an actively bad strategy rather than a merely neutral one,
    directly countering the "do nothing, stay upright, collect a free
    per-step bonus" local optimum alive_bonus used to create.

    min_speed=0.03 m/s (3 cm/s) is an unverified starting guess for "this
    barely counts as moving" -- tune based on what speeds actually show up
    once training gets past standing-still (if it converges to always
    exceeding this trivially, raise it; if it never manages to escape
    the penalty, lower it).
    """
    forward_vel = _root_lin_vel_w(env, asset_cfg)[:, 0]
    shortfall = torch.clamp(min_speed - torch.abs(forward_vel), min=0.0)
    return torch.square(shortfall)


def fallen_over(env, asset_cfg: SceneEntityCfg, upright_threshold: float = 0.5) -> torch.Tensor:
    """Termination: true once tipped far enough that projected gravity's
    body-frame z-component drops below the threshold in magnitude.
    upright_threshold=0.5 is an unvalidated starting guess; tune from play."""
    grav_b = projected_gravity(env, asset_cfg)
    return torch.abs(grav_b[:, 2]) < upright_threshold


def _build_rewards() -> dict:
    return {
        "forward_progress": RewardTermCfg(
            func=forward_progress_reward, weight=5.0, params={"asset_cfg": root_cfg},
        ),
        "idle_penalty": RewardTermCfg(
            func=idle_penalty, weight=-2.0, params={"asset_cfg": root_cfg, "min_speed": 0.0005},
        ),
        "excess_speed": RewardTermCfg(
            func=excess_speed_penalty, weight=-1.0,
            params={"asset_cfg": root_cfg, "max_speed": _MAX_PLAUSIBLE_SPEED_MS},
        ),
        "upright": RewardTermCfg(func=upright_penalty, weight=-1.0, params={"asset_cfg": root_cfg}),
        "lateral_drift": RewardTermCfg(func=lateral_drift_penalty, weight=-0.5, params={"asset_cfg": root_cfg}),
        "action_rate": RewardTermCfg(func=mdp_rewards.action_rate_l2, weight=-0.001, params={}),
    }


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    #"fallen_over": TerminationTermCfg(
    #    func=fallen_over, params={"asset_cfg": root_cfg, "upright_threshold": 0.5},
    #),
}

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
events = {
    "reset_root": EventTermCfg(
        func=reset_root_state_uniform,  # UNVERIFIED, see module docstring
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            "asset_cfg": root_cfg,
        },
    ),
    "reset_joints": EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": arm_cfg,
        },
    ),
}


def get_crawler_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"crawler": _get_robot_cfg()},
            num_envs=1 if play else 1024,
            env_spacing=0.5,
        ),
        commands={},  # no command term -- reward maximizes raw forward
                      # speed directly, nothing to track. UNVERIFIED: if
                      # ManagerBasedRlEnvCfg requires at least one command
                      # term, this will need a no-op placeholder instead.
        observations=observations,
        actions=actions,
        events=events,
        rewards=_build_rewards(),
        terminations=terminations,
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01)),
        decimation=10,  # 10 * 0.01 = 0.1s/step = 10 Hz, matching real motor cadence
        episode_length_s=15.0,
    )