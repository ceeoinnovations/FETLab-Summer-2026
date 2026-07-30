import torch.nn as nn
from torchvision import models

# Symbol class names — must stay alphabetically sorted to match ImageFolder.
SYMBOL_CLASSES = ["backward", "forward", "left", "right", "stop"]

# Motor commands for each symbol: (left_speed, right_speed)
MOTOR_MAP = {
    "forward":  ( 100,  100),
    "backward": (-100, -100),
    "left":     ( -70,  100),
    "right":    ( 100,  -70),
    "stop":     (   0,    0),
}


def build_model(num_classes: int):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m
