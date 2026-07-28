"""
Offline RL actor (TD3+BC) for the vision-based target-seeking task.

Takes the compact 4-dim state (cx_norm, cy_norm, area_frac, visible) —
not an image — and outputs [left_speed, right_speed] as a normalized
(-ACTION_SCALE..ACTION_SCALE) action. See config.py for why this operates
on the compact state instead of a learned image backbone, and why
ACTION_SCALE is baked in here rather than left as a caller-side choice —
it must match exactly what a trained checkpoint's weights assume.
"""
import torch
import torch.nn as nn

from config import STATE_DIM, ACTION_DIM, ACTION_SCALE, HIDDEN_SIZE


def build_offline_rl_actor():
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, HIDDEN_SIZE), nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
                nn.Linear(HIDDEN_SIZE, ACTION_DIM), nn.Tanh(),
            )

        def forward(self, s):
            return self.net(s) * ACTION_SCALE

    return Actor()


class Critic(nn.Module):
    """Twin critics are just two independent instances of this — used only
    during training (train_offline_rl.py), never at inference/deployment
    time, so it isn't exposed through build_offline_rl_actor()."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM + ACTION_DIM, HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=1))
