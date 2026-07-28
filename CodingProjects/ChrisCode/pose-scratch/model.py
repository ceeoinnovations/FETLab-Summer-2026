import torch.nn as nn

# Gesture class names — must stay alphabetically sorted to match ImageFolder.
GESTURE_CLASSES = ["backward", "forward", "stop", "turn_left", "turn_right"]


class ScratchCNN(nn.Module):
    """
    A small convolutional network trained entirely from random weights.
    No pretrained backbone — every parameter is learned from your images.

    Architecture (input: 224×224×3):
      Conv block 1: 3 → 32 channels,  224 → 112
      Conv block 2: 32 → 64 channels, 112 →  56
      Conv block 3: 64 → 128 channels,  56 →  28
      Conv block 4: 128 → 128 channels, 28 →  14
      Global average pool:              14 →   1
      FC head: 128 → 64 → num_classes
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # block 1
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 3
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # block 4
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # squeeze spatial dims to 1×1
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(num_classes: int) -> ScratchCNN:
    return ScratchCNN(num_classes)
