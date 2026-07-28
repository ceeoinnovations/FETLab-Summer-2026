"""
Perception model: predicts where the target is and how big it looks,
instead of predicting motor speeds directly (see model.py for the old
approach).

IMPORTANT ARCHITECTURE NOTE: torchvision's mobilenet_v2 unconditionally
average-pools its feature map to a single vector BEFORE its classifier
ever runs (see MobileNetV2._forward_impl — the pooling is baked into the
model's forward(), not something you can skip just by replacing
`.classifier`). Reusing `backbone.classifier = nn.Sequential(...)` the
way model.py's original motor-speed head did would silently keep that
pooling in place, throwing away exactly the left/right position
information this model needs — the same bug that caused the original
project's wrong-direction-turn problem. This file avoids that by using
`backbone.features` directly (which is NOT pooled) and building a custom
forward pass:

  - cx_norm, cy_norm : extracted via spatial softmax over the backbone's
    raw 7x7 feature map — this is the part that must never be pooled,
    since position is exactly the information pooling would destroy.
  - area_frac, visible_logit : extracted from an ordinary pooled vector
    through a small FC head — pooling is fine here, since "how big" and
    "is it there at all" don't depend on knowing WHERE the way steering
    direction does.
"""

import torch
import torch.nn as nn
from torchvision import models


class DetectorModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features  # (B, 1280, 7, 7) for a 224x224 input — NOT pooled

        # Trainable squeeze: reduces 1280 channels to 1, keeping the 7x7
        # spatial layout intact. This is the only path cx_norm/cy_norm can
        # come from, and it never gets collapsed by pooling.
        self.position_squeeze = nn.Conv2d(1280, 1, kernel_size=1)

        # Separate small head for the two properties that don't need
        # precise location — ordinary global-average-pooled features are
        # fine here.
        self.global_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [area_raw, visible_logit]
        )

    def forward(self, x):
        feat = self.features(x)  # (B, 1280, H, W) — H=W=7 for a 224x224 input
        B, C, H, W = feat.shape

        # --- position: spatial softmax, never pooled ---
        position_map = self.position_squeeze(feat).view(B, H * W)   # (B, H*W)
        weights = torch.softmax(position_map, dim=1).view(B, H, W)   # sums to 1 over space
        xs = torch.linspace(-1, 1, W, device=x.device).view(1, 1, W)
        ys = torch.linspace(-1, 1, H, device=x.device).view(1, H, 1)
        cx_norm = (weights * xs).sum(dim=(1, 2))  # (B,) — probability-weighted x position
        cy_norm = (weights * ys).sum(dim=(1, 2))  # (B,) — kept for completeness, unused downstream

        # --- area / visibility: ordinary pooling is fine here ---
        pooled = nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 1280)
        area_raw, visible_logit = self.global_head(pooled).unbind(dim=1)
        area_frac = torch.sigmoid(area_raw)

        return cx_norm, cy_norm, area_frac, visible_logit
