"""
Model definition for behavior-cloning steering.

Architecture: MobileNetV2 backbone (frozen, ImageNet pretrained) with
a small regression head outputting two continuous values:
    [left_speed, right_speed]   both normalized to roughly -1..1

Same backbone-freezing strategy as the symbol classifier — the
backbone already understands edges/shapes, so only the head needs
training on a modest dataset.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model():
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(backbone.last_channel, 64),
        nn.ReLU(),
        nn.Linear(64, 2),   # [left_speed, right_speed], normalized -1..1
        nn.Tanh(),          # keeps output bounded in -1..1
    )
    return backbone
