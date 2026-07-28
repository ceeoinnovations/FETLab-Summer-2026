"""
Perception model for Option 3: sorts each frame into one of a fixed list
of categories (see config.CATEGORIES) instead of predicting any kind of
position. Standard image classification — same MobileNetV2 backbone as
the other options, ordinary global average pooling (no special
spatial-preserving trick), a small fully-connected head ending in one
score per category.

Unlike Options 1 and 2, there is no continuous number anywhere in this
model's output — the entire point is a coarse, discrete decision. See
config.CATEGORIES for what "coarse" means concretely here, and drive.py
for how a category becomes a motor command (a fixed lookup table, not a
computed formula).
"""

import torch.nn as nn
from torchvision import models
from config import CATEGORIES


def build_classifier_model():
    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    backbone.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(backbone.last_channel, 64),
        nn.ReLU(),
        nn.Linear(64, len(CATEGORIES)),  # one raw score per category — softmax applied by the loss/at inference
    )
    return backbone
