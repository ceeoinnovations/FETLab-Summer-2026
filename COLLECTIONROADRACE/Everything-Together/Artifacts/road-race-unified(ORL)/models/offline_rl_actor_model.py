"""
Offline RL actor (TD3+BC) for the vision-based target-seeking task.

Unlike every other model in this project, this one does NOT take an
image as input. It operates on a compact 4-dim visual state — cx_norm,
cy_norm, area_frac, visible — the same state the color-threshold
detector (color_detect.py) produces. That choice wasn't stylistic: an
earlier attempt to learn dynamics/value functions over the full
MobileNetV2 feature vector (1280-dim) overfit badly given the amount
of training data available; collapsing to this 4-dim task-relevant
state was what made training stable at all.

Trained via TD3+BC (behavior-cloning-regularized offline RL) on human
joystick driving data (data.zip + the original road-race-end-to-end
project's demonstration data) — NOT on any of this project's autonomous
(color/keypoint/grid-detector-driven) data, which turned out to lack
the "same visual situation, different action" contrast offline RL
needs; see project notes for the full investigation.

Closed-loop evaluation (via a learned dynamics-model proxy) showed this
actor performing on par with, not better than, a plain behavior-cloned
model trained on the same data — included here anyway so it can be
compared directly on real hardware, which is the one test the proxy
evaluation couldn't fully settle.

ACTION_SCALE=0.2 is not a tunable knob — it must match the scale baked
into the actor at training time (its output was tanh(...)*0.2, matching
the training data's raw-motor-command range of roughly -17..17 once
denormalized by *100 downstream). Changing it changes what the
checkpoint's weights actually mean.
"""
import torch.nn as nn

ACTION_SCALE = 0.2


def build_offline_rl_actor():
    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(4, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, 2), nn.Tanh(),
            )

        def forward(self, s):
            return self.net(s) * ACTION_SCALE

    return Actor()

