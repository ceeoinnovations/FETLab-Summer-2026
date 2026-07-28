"""
Learned dynamics model for standalone model-predictive control (mode 0).

Predicts how the compact 4-dim state (cx_norm, cy_norm, area_frac,
visible) changes given an action: (state, action) -> delta, where the
next state is state + delta. Operates on the SAME compact state as the
offline RL actor (mode 9), for the same reason — see that model's
docstring: a full learned visual feature vector overfit badly given the
amount of training data available, while this 4-dim task-relevant state
trained stably.

Trained on human joystick driving data (the same data mode 9's actor was
trained on) — self-supervised, no reward or demonstrator judgment
involved, just "what actually happened next" given what was done. This is
what makes it usable as an "imagined future" for planning: nothing about
its own training required a policy to imitate or a reward to maximize,
only real recorded transitions.

Validated (see project notes) to modestly but genuinely beat a naive
"assume nothing changes" baseline specifically at the 3-8 step horizons
planning cares about — weaker or roughly tied with that baseline at 1-2
steps. This is why MODE0_HORIZON defaults to 5 rather than something
shorter: the model's edge over doing nothing only shows up a few steps
out.
"""
import torch
import torch.nn as nn


def build_mpc_dynamics_model():
    class Dynamics(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(6, 32), nn.ReLU(),
                nn.Linear(32, 32), nn.ReLU(),
                nn.Linear(32, 4),
            )

        def forward(self, state, action):
            return self.net(torch.cat([state, action], dim=1))

    return Dynamics()
