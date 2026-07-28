import torch.nn as nn
from torchvision import models

# Gesture class names — these become the subfolder names under data/
# ImageFolder sorts them alphabetically, so this list must stay sorted too.
GESTURE_CLASSES = ["backward", "forward", "stop", "turn_left", "turn_right"]


def build_model(num_classes: int):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m
