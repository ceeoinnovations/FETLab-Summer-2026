from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ForwardVelocityCommand(CommandTerm):
    """Scalar target forward (world +x) speed for the crawler to track.

    Simpler than ReachPositionCommand's fixed/dynamic/hemisphere modes
    since there's only one axis worth commanding here -- this robot has no
    steering DOF (just two flapping arms), so there's no meaningful
    "direction" command to give it, only "how fast forward."
    """

    cfg: ForwardVelocityCommandCfg

    def __init__(self, cfg: ForwardVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.asset: Entity = env.scene[cfg.entity_name]

        self.target_velocity = torch.zeros(self.num_envs, 1, device=self.device)

        self.metrics["velocity_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_velocity

    def _update_metrics(self) -> None:
        # CONFIRMED against installed mjlab's entity/data.py:
        # root_link_lin_vel_w is a flat [B, 3] per-env world-frame velocity,
        # no body_ids indexing needed (unlike the geom-level fields we
        # mistakenly tried earlier).
        forward_vel = self.asset.data.root_link_lin_vel_w[:, 0]
        self.metrics["velocity_error"] = torch.abs(self.target_velocity.squeeze(-1) - forward_vel)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        lower = torch.tensor([self.cfg.velocity_range[0]], device=self.device)
        upper = torch.tensor([self.cfg.velocity_range[1]], device=self.device)
        self.target_velocity[env_ids] = sample_uniform(lower, upper, (n, 1), device=self.device)

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class ForwardVelocityCommandCfg(CommandTermCfg):
    entity_name: str
    velocity_range: tuple[float, float] = (0.05, 0.3)

    def build(self, env: ManagerBasedRlEnv) -> ForwardVelocityCommand:
        return ForwardVelocityCommand(self, env)