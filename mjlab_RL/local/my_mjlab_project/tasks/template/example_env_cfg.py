"""
TEMPLATE task env_cfg -- structure distilled from two working mjlab
tasks (a fixed-base arm reach task, a free-base velocity-controlled
crawler). This is meant as a checklist/skeleton for a new task, not
something to run as-is -- every TODO needs filling in for your specific
robot and objective.

=====================================================================
THINGS WORTH VERIFYING AGAINST YOUR INSTALLED MJLAB VERSION BEFORE
TRUSTING ANY OF THIS, RATHER THAN ASSUMING IT MATCHES WHAT'S BELOW:

  - Field names on entity.data (e.g. root_link_quat_w, root_link_lin_vel_w,
    root_link_lin_vel_b, projected_gravity_b, joint_pos, joint_vel).
    Confirmed for us via:
      findstr /n "def " path\to\mjlab\entity\data.py
    then pasting the relevant property definitions.
  - Available action term configs/classes (JointPositionActionCfg,
    JointVelocityActionCfg, RelativeJointPositionActionCfg, etc.) and
    their exact fields. Confirmed via:
      findstr /n "class " path\to\mjlab\envs\mdp\actions\actions.py
      type path\to\mjlab\envs\mdp\actions\__init__.py
    IMPORTANT gotcha we hit: `clip` on action cfgs is NOT a plain
    (min, max) tuple -- it goes through the same per-actuator-name
    dict resolution as `scale`/`offset`. Use
    `clip={name: (min, max) for name in actuator_names}`, not a bare
    tuple, or you'll get a TypeError from resolve_matching_names_values.
  - Available observation functions in mjlab.envs.mdp.observations
    (joint_pos_rel, joint_vel_rel, projected_gravity, base_lin_vel,
    base_ang_vel, last_action, generated_commands, builtin_sensor, etc.)
    and their exact signatures. Confirmed via the same findstr pattern
    against mjlab/envs/mdp/observations.py.
  - Whether ManagerBasedRlEnvCfg accepts `commands={}` when a task has
    no command term at all (we assumed yes; never independently
    confirmed against the source).
  - Reset event functions in mjlab.envs.mdp.events -- only
    reset_joints_by_offset was directly confirmed via import in a
    working file; reset_root_state_uniform (needed for any free-base
    robot) was a guessed sibling name, never grepped/confirmed.
=====================================================================

DESIGN DECISIONS TO MAKE FOR A NEW TASK (roughly the order we worked
through them in practice):

  1. ACTIONS: What does the real hardware's command interface actually
     look like? Position target? Velocity/speed command? This should
     drive the action term choice (JointPositionActionCfg /
     RelativeJointPositionActionCfg / JointVelocityActionCfg / a custom
     wrapper), not the other way around -- match the real interface,
     then figure out sim training from there. Also consider:
       - Does the real actuator have a deadzone (won't move below some
         command magnitude due to static friction)? If yes, train
         against that deadzone (see DeadzoneAction pattern below) or
         the policy will learn fine control the hardware can't deliver.
       - Does it need a hard safety clip matching a real physical limit
         (torque, velocity, RPM)? Convert the real spec into sim units
         explicitly and comment where the number came from.

  2. OBSERVATIONS: What can the real robot ACTUALLY sense at
     deployment? Anything the actor observation group includes must be
     reconstructable from real sensors, or the trained policy simply
     cannot run on hardware. Split into two groups:
       - actor: only real-sensor-derivable signals (joint encoders,
         onboard IMU, commanded targets, previous action).
       - critic: actor's signals PLUS privileged sim-only ground truth
         (e.g. true world-frame velocity, if there's no real velocity
         sensor) -- the critic is discarded at deployment, so it's free
         to see things that make training easier/faster without
         breaking sim2real transfer.
     This distinction only matters if the two differ; if every signal
     you want is realistically available, actor and critic can be
     identical and there's no need to overengineer a split.

  3. REWARDS: Start from "what's the one thing this task actually
     needs to optimize," and be suspicious of any reward term that's
     positive/free regardless of whether the desired behavior happens
     (that's exactly how we hit a "stand still and collect free reward"
     local optimum in practice -- an unconditional alive_bonus). Prefer:
       - one primary objective term with real weight
       - penalty terms that are ZERO in the desired regime and only
         activate outside it (hinge-shaped: torch.clamp(x - limit, min=0)
         squared), so they don't fight the primary objective within the
         acceptable range
       - be wary of hinge penalties whose "safe zone" boundary becomes
         a new target the policy settles for and stops improving past
         (we saw a policy converge to sit exactly at an idle-penalty
         threshold rather than continuing to optimize the primary
         reward past it) -- set such thresholds as trivial floors, not
         performance targets.

  4. TERMINATIONS/EVENTS: time_out is almost always wanted. Add
     failure-condition terminations (e.g. tipped over) if premature
     episode-ending on failure will help training; note this trades off
     against not learning recovery behavior -- consider commenting it
     out during a debugging pass and re-enabling once behavior is sane
     (one of our two examples did exactly this).
=====================================================================
"""

import math
from pathlib import Path
from dataclasses import dataclass

import mujoco
import torch

from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
from mjlab.actuator import XmlActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointVelocityActionCfg  # TODO: or JointPositionActionCfg / RelativeJointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointVelocityAction  # TODO: match the above
from mjlab.envs.mdp.observations import (
    joint_pos_rel,
    joint_vel_rel,
    projected_gravity,
    last_action,
    base_lin_vel,
    # TODO: add generated_commands if you have a command term, base_ang_vel
    # if useful, builtin_sensor for anything wired through a MuJoCo <sensor>
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
from mjlab.envs.mdp.events import reset_joints_by_offset  # TODO: + reset_root_state_uniform if free-base, VERIFY first

# TODO: only needed if you have a command term -- see commands.py template
# and the "when do you even need a command" note there.
# from .commands import TemplateCommandCfg

_ROBOT_XML = Path(__file__).parent / "robot.xml"
_JOINT_NAMES = ("joint_1",)  # TODO: all actuated joint names, in a fixed order

_ENTITY_NAME = "robot"  # TODO: pick a name; used throughout as the scene key

# ---------------------------------------------------------------------------
# Real hardware spec constants -- fill these in from actual datasheet/
# measured values, and comment where each number came from. Anything
# marked "placeholder" below should get a matching comment explaining
# it's unverified and where to get the real number.
# ---------------------------------------------------------------------------
_CONTROL_HZ = 10.0  # TODO: match your real control loop rate
_CONTROL_STEP_S = 1.0 / _CONTROL_HZ


def _get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(_ROBOT_XML))
    spec.compiler.meshdir = str(_ROBOT_XML.parent / "assets")
    # TODO: FREE-BASE ONLY, if you didn't add <freejoint/> directly in
    # the XML: spec.body("root_body_name").add_freejoint()
    return spec


_ROBOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(XmlActuatorCfg(target_names_expr=tuple(_JOINT_NAMES)),),
)

_ROBOT_INIT = EntityCfg.InitialStateCfg(
    # TODO: FREE-BASE ONLY -- pos/rot/lin_vel/ang_vel fields apply to the
    # freejoint root. Spawn height is easy to get wrong; tune by watching
    # `uv run play` for floor-clipping vs. floating.
    # pos=(0.0, 0.0, 0.05),
    # rot=(1.0, 0.0, 0.0, 0.0),
    # lin_vel=(0.0, 0.0, 0.0),
    # ang_vel=(0.0, 0.0, 0.0),
    joint_pos={name: 0.0 for name in _JOINT_NAMES},
    joint_vel={name: 0.0 for name in _JOINT_NAMES},
)


def _get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=_ROBOT_ARTICULATION,
        init_state=_ROBOT_INIT,
    )


arm_cfg = SceneEntityCfg(_ENTITY_NAME, joint_names=_JOINT_NAMES)
# FREE-BASE ONLY: root-state accessors (root_link_quat_w etc.) read
# directly off the entity, no body_names/body_ids resolution needed.
root_cfg = SceneEntityCfg(_ENTITY_NAME)
# FIXED-BASE, TRACKING A SITE (e.g. end-effector): resolves to site_ids.
# ee_cfg = SceneEntityCfg(_ENTITY_NAME, site_names=("tracked_point",))


# ---------------------------------------------------------------------------
# World-frame root state helpers -- FREE-BASE ONLY. No built-in mjlab
# observation function exposes world-frame velocity (by design: it's
# sim-only/privileged) -- use these in reward functions and the critic
# observation group only, never the actor.
# ---------------------------------------------------------------------------
def _root_quat_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_quat_w


def _root_lin_vel_w(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.root_link_lin_vel_w


# TODO: FIXED-BASE, TRACKING A SITE -- pattern for a custom "vector from
# end-effector to target" observation/reward function:
# def ee_to_target(env, asset_cfg: SceneEntityCfg, command_name: str) -> torch.Tensor:
#     asset = env.scene[asset_cfg.name]
#     ee_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
#     target_pos_w = env.command_manager.get_command(command_name)
#     return target_pos_w - ee_pos_w


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
# TODO: if the real hardware has a command deadzone (won't move below
# some magnitude due to static friction), subclass whichever action you
# chose and zero out sub-deadzone commands in apply_actions(), following
# this pattern (shown here for JointVelocityAction; the same idea
# applies to any action type):
#
# @dataclass(kw_only=True)
# class DeadzoneJointVelocityActionCfg(JointVelocityActionCfg):
#     deadzone_fraction: float = 0.10
#     def build(self, env) -> "DeadzoneJointVelocityAction":
#         return DeadzoneJointVelocityAction(self, env)
#
# class DeadzoneJointVelocityAction(JointVelocityAction):
#     cfg: DeadzoneJointVelocityActionCfg
#     def apply_actions(self) -> None:
#         command_fraction = self._processed_actions / self.cfg.scale
#         below_deadzone = torch.abs(command_fraction) < self.cfg.deadzone_fraction
#         velocity_target = torch.where(
#             below_deadzone, torch.zeros_like(self._processed_actions), self._processed_actions
#         )
#         self._entity.set_joint_velocity_target(velocity_target, joint_ids=self._target_ids)

actions = {
    "TODO_action_term_name": JointVelocityActionCfg(
        entity_name=_ENTITY_NAME,
        actuator_names=_JOINT_NAMES,
        scale=1.0,  # TODO: real units conversion -- e.g. for velocity
                    # control, this is "action=1.0 -> what rad/s", derived
                    # from a real max-speed spec.
        offset=0.0,
        use_default_offset=False,
        # `clip` MUST be a dict, not a bare tuple -- see module docstring.
        clip={name: (-1.0, 1.0) for name in _JOINT_NAMES},  # TODO: real limit
    ),
}

# ---------------------------------------------------------------------------
# Observations -- see module docstring's design-decision #2 for the
# actor/critic split rationale.
# ---------------------------------------------------------------------------
actor_terms = {
    "joint_pos": ObservationTermCfg(func=joint_pos_rel, params={"asset_cfg": arm_cfg}),
    "joint_vel": ObservationTermCfg(func=joint_vel_rel, params={"asset_cfg": arm_cfg}),
    # TODO: FREE-BASE -- orientation/balance sensing, if the real robot
    # has an IMU:
    # "projected_gravity": ObservationTermCfg(func=projected_gravity, params={"asset_cfg": root_cfg}),
    "last_action": ObservationTermCfg(func=last_action, params={"action_name": None}),
    # TODO: if you have a command term:
    # "target": ObservationTermCfg(func=generated_commands, params={"command_name": "TODO"}),
}

critic_terms = {
    **actor_terms,
    # TODO: privileged sim-only signals unavailable on real hardware,
    # e.g. true velocity if there's no real velocity/position sensor:
    # "base_lin_vel": ObservationTermCfg(func=base_lin_vel, params={"asset_cfg": root_cfg}),
}

observations = {
    "actor": ObservationGroupCfg(actor_terms),
    "critic": ObservationGroupCfg(critic_terms),
}

# ---------------------------------------------------------------------------
# Rewards -- see module docstring's design-decision #3.
# ---------------------------------------------------------------------------
def primary_objective_reward(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """TODO: the one thing this task is actually trying to optimize."""
    raise NotImplementedError


def _build_rewards() -> dict:
    return {
        "primary_objective": RewardTermCfg(
            func=primary_objective_reward, weight=1.0, params={"asset_cfg": root_cfg},
        ),
        "action_rate": RewardTermCfg(func=mdp_rewards.action_rate_l2, weight=-0.01, params={}),
        # TODO: add task-specific shaping terms (upright, drift, idle,
        # excess-speed safety caps, etc.) following the hinge-shaped
        # "zero in the safe zone" pattern described in the module
        # docstring, rather than an unconditional positive bonus.
    }


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------
terminations = {
    "time_out": TerminationTermCfg(func=time_out, time_out=True),
    # TODO: failure-condition termination, e.g. tipped over. Consider
    # commenting this out during initial debugging and re-enabling once
    # basic behavior looks sane.
}

# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
events = {
    "reset_joints": EventTermCfg(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": arm_cfg,
        },
    ),
    # TODO: FREE-BASE -- reset_root_state_uniform or equivalent, VERIFY
    # the real function name/signature first (see module docstring).
}


def get_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={_ENTITY_NAME: _get_robot_cfg()},
            num_envs=8 if play else 1024,
            env_spacing=1.0,
        ),
        commands={},  # TODO: replace with your command dict if you have one
        observations=observations,
        actions=actions,
        events=events,
        rewards=_build_rewards(),
        terminations=terminations,
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.01)),
        decimation=10,  # decimation * timestep = control step; match your real loop rate
        episode_length_s=15.0,
    )