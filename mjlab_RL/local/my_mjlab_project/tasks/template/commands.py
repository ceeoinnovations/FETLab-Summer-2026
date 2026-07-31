"""
TEMPLATE command term -- structure distilled from two working examples:
a 3D position target (ReachPositionCommand, fixed/dynamic/hemisphere
sampling modes) and a scalar target-speed command (ForwardVelocityCommand).

WHEN DO YOU EVEN NEED A COMMAND?
A command term is for anything the environment should tell the policy to
achieve that changes across (or within) episodes -- "go to this point",
"move at this speed", "reach this joint configuration". If your task has
a single, fixed objective for every episode (e.g. "just go as fast as
possible, always"), you likely don't need a command term at all.
Don't add one speculatively, only if needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class TemplateCommand(CommandTerm):
    """
    TODO: rename and describe what this command represents.

    Required overrides (this is the full CommandTerm contract we've
    exercised in practice -- there may be more optional hooks available,
    but these are the ones both working examples needed):
      - __init__: set up any persistent per-env state tensors + metrics
      - command (property): return the current command tensor
      - _update_metrics: called each step, update self.metrics for logging
      - _resample_command: called for env_ids that need a new command
        (periodically per resampling_time_range, and on reset)
      - _update_command: called every step regardless of resampling;
        use this for anything the command needs to do continuously
        (e.g. re-deriving a value from current state). Often a no-op.
      - _debug_vis_impl: optional, draws something in the viewer via
        `uv run play`'s debug visualization -- skip if you don't need it.
    """

    cfg: TemplateCommandCfg

    def __init__(self, cfg: TemplateCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.asset: Entity = env.scene[cfg.entity_name]

        # TODO: size this to your actual command dimensionality (3 for a
        # position target, 1 for a scalar speed target, etc.)
        self.command_value = torch.zeros(self.num_envs, 1, device=self.device)

        # Anything you want logged/plotted during training goes in
        # self.metrics -- both examples tracked at least an error metric.
        self.metrics["error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_value

    def _update_metrics(self) -> None:
        # TODO: compute whatever error/success signal makes sense here,
        # e.g. distance from a target, or |target - actual|.
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)

        # TODO: replace with your actual sampling logic. sample_uniform
        # is the confirmed-available helper for uniform ranges; for
        # anything more structured (e.g. uniform-in-volume sphere
        # sampling), see the hemisphere-mode example this was distilled
        # from for a rejection-sampling pattern.
        lower = torch.tensor([0.0], device=self.device)
        upper = torch.tensor([1.0], device=self.device)
        self.command_value[env_ids] = sample_uniform(lower, upper, (n, 1), device=self.device)

    def _update_command(self) -> None:
        # Often a no-op (both examples left this empty) -- only needed
        # if the command has to be re-derived every step rather than
        # just periodically resampled.
        pass

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        # Optional. Example from the position-target command:
        #   env_indices = visualizer.get_env_indices(self.num_envs)
        #   for batch in env_indices:
        #       visualizer.add_sphere(center=..., radius=0.02, color=..., label=...)
        pass


@dataclass(kw_only=True)
class TemplateCommandCfg(CommandTermCfg):
    entity_name: str

    # TODO: add whatever fields your sampling logic needs -- ranges,
    # difficulty modes, success thresholds, etc. The position-target
    # example used a Literal["fixed", "dynamic", "hemisphere"] difficulty
    # field with a nested dataclass per mode; only add that complexity if
    # you actually need multiple sampling strategies.

    def build(self, env: ManagerBasedRlEnv) -> TemplateCommand:
        return TemplateCommand(self, env)