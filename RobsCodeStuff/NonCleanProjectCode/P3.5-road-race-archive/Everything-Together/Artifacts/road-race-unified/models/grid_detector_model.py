"""
A simplified, single-class, single-object detector, built around the same
core idea as YOLO — divide the image into a grid, let each cell predict
whether an object's center is there plus that object's box — without the
extra machinery a general-purpose multi-class, multi-object, multi-scale
detector like real YOLO needs (anchor boxes, multiple detection scales, a
class-probability vector per cell). Since this project only ever has one
object type and expects at most one instance in frame, that machinery
would add real complexity for no benefit.

Architecture: the frozen backbone's last conv layer already produces a
7x7 spatial grid for a 224x224 input (MobileNetV2 downsamples by a factor
of 32 = 224/32 = 7) — this IS the grid, we don't build one separately.
A trainable 1x1 convolution reduces the backbone's 1280 channels straight
down to 5 per cell:
  - confidence : is the object's center in this cell?
  - x_offset, y_offset : where in this cell, precisely (0..1 within the cell)
  - w, h : the box's width/height as a fraction of the WHOLE frame

Unlike detector_model.py (Option 1), this deliberately does NOT use
spatial softmax or global average pooling anywhere — every cell keeps and
reports its own independent prediction, which is what lets one frame
describe "object here, this size" rather than only "object somewhere,
roughly this size" (spatial softmax) or a single number with no space at
all (global average pooling).
"""

import torch
import torch.nn as nn
from torchvision import models

GRID_SIZE = 7  # MobileNetV2's natural output resolution at 224x224 input


class GridDetector(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = backbone.features  # (B, 1280, 7, 7) for a 224x224 input
        self.head = nn.Conv2d(1280, 5, kernel_size=1)  # per-cell: confidence, x, y, w, h

    def forward(self, x):
        feat = self.features(x)             # (B, 1280, 7, 7)
        raw = self.head(feat)                # (B, 5, 7, 7)
        confidence = torch.sigmoid(raw[:, 0])  # (B, 7, 7) — is the object's center here?
        x_offset = torch.sigmoid(raw[:, 1])    # (B, 7, 7) — 0..1 position within the cell
        y_offset = torch.sigmoid(raw[:, 2])
        w = torch.sigmoid(raw[:, 3])            # (B, 7, 7) — 0..1 fraction of the whole frame
        h = torch.sigmoid(raw[:, 4])
        return confidence, x_offset, y_offset, w, h


def decode_prediction(confidence, x_offset, y_offset, w, h, confidence_threshold=0.5):
    """Turn one image's raw grid output into a single best detection (or
    None). Since we expect at most one real object, this just takes the
    highest-confidence cell — no need for the non-max-suppression step a
    multi-object detector like real YOLO requires to remove duplicates.

    All inputs are (GRID_SIZE, GRID_SIZE) tensors for a single image
    (batch already stripped). Returns a dict with cx_norm, cy_norm,
    area_frac, bbox (pixel coords need frame width/height — see detect.py)
    or None if the best cell's confidence is below the threshold.
    """
    flat_idx = torch.argmax(confidence)
    row = int(flat_idx // GRID_SIZE)
    col = int(flat_idx % GRID_SIZE)
    best_confidence = float(confidence[row, col])
    if best_confidence < confidence_threshold:
        return None

    cx_frac = (col + float(x_offset[row, col])) / GRID_SIZE
    cy_frac = (row + float(y_offset[row, col])) / GRID_SIZE
    w_frac = float(w[row, col])
    h_frac = float(h[row, col])

    return {
        "cx_norm": cx_frac * 2 - 1,
        "cy_norm": cy_frac * 2 - 1,
        "area_frac": w_frac * h_frac,
        "cx_frac": cx_frac,
        "cy_frac": cy_frac,
        "w_frac": w_frac,
        "h_frac": h_frac,
        "confidence": best_confidence,
        "cell": (row, col),
    }
