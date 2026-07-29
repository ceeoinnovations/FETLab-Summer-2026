# Option 2 — Object detection (simplified, YOLO-style)

Perception method: a small trained network divides each frame into a 7x7
grid and predicts, for every cell, whether the target's center is there
plus its full bounding box — a simplified, single-class, single-object
version of YOLO's core idea (see `grid_detector_model.py`'s docstring for
what was deliberately left out and why). Steering and speed are decided
by the same hand-written proportional controller as the other two
options — this only replaces how the centroid/size get found.

## Hardware & environment setup (do this first)

This assumes the same physical setup as your original road-race project —
nothing hardware-related changed here, only the perception method.

**1. Install Python dependencies:**
```
pip install torch torchvision opencv-python pillow numpy matplotlib
```

**2. The hardware values in `config.py` are already filled in from your
original setup** — double-check they still match before running anything:
- `SERIAL = 1227` — the LEGO Bluetooth card serial for your motor +
  joystick controller (printed on the card itself; re-check if you've
  swapped cards since).
- `SERIAL_COLOR_SENSOR = 2283` — the separate card for the color sensor.
  This must stay a different card from `SERIAL` — pairing the color
  sensor to the same card as the motor makes the hub apply its own
  built-in reflex on top of whatever the Python code commands.
- `CAMERA = 1` — the index Camo Studio's virtual webcam appears as. If a
  script can't open the camera, try `0` (laptop webcam), or confirm the
  index with a minimal `cv2.VideoCapture(n)` test.

**3. Before running `collect_data.py`, `calibrate_color.py`, or `drive.py`:**
- Camo Studio must already be running with your phone connected — these
  scripts fail immediately if the camera isn't available yet.
- The motor/controller hub and the (separately-carded) color sensor need
  to be powered on and in range. `connect()` retries a few times if a
  card reports "not ready," but won't wait forever.

**4. Physical/environment notes:**
- Keep other objects matching your target's color out of frame — the
  color detector (used directly, or to generate training labels) can't
  distinguish your target from anything else in the same HSV range.
- Run `calibrate_color.py` in the actual room/lighting you'll operate
  in; HSV values tuned elsewhere often don't transfer.

## Setup order

Each script below takes two command-line arguments (an images folder and
an output CSV). If you run a script with no arguments — e.g. by pressing
VS Code's Run button rather than typing a command in a terminal — it
falls back to sensible defaults (`data/images` and `data/pseudo_*.csv`,
resolved next to the script itself) and prints which paths it picked.
Terminal arguments still work exactly as before if you want to override
them (e.g. pointing at a different dataset).

1. `python collect_data.py` — drive the car around, collecting images +
   session markers.
2. `python calibrate_color.py` — tune HSV_LOWER/HSV_UPPER in `config.py`.
   This is a bootstrapping step: the grid detector's training labels come
   FROM this color detector's bounding boxes.
3. `python generate_pseudo_boxes.py data/images data/pseudo_boxes.csv` —
   runs the tuned color detector over every image and records its full
   bounding box (not just a centroid) as the training label. Spot-check a
   random sample against the source images before trusting it.
4. `python train_grid_detector.py data/images data/pseudo_boxes.csv` —
   trains `grid_detector_model.py`. Produces `grid_detector_model.pt`.
5. `python drive.py` — runs autonomously using the trained grid detector
   (`config.DETECTOR_BACKEND` defaults to `"grid"` in this project).

## Files

- `grid_detector_model.py` — the detector network: frozen MobileNetV2
  backbone (spatial output kept, not pooled away) + a small trainable
  1x1 convolution predicting confidence + box per grid cell
- `generate_pseudo_boxes.py` — turns existing images into box labels
  using the color detector (always uses the color method, regardless of
  `DETECTOR_BACKEND`)
- `train_grid_detector.py` — trains the model with a combined confidence
  + box-regression loss, the box term only counted at the one cell
  responsible for the real object
- `detect.py` — `get_target()` dispatches to color or grid based on
  `config.DETECTOR_BACKEND`; `get_target_color()` always uses color
- `drive.py` — unchanged from the other two options; it only depends on
  `get_target()`'s output shape, not how it was produced
- `calibrate_color.py`, `collect_data.py`, `config.py`, `lelib.py` —
  shared with the other two options

## A note on training this yourself

I validated this entire pipeline end-to-end against real project data —
label generation, the grid/cell-responsibility math, the masked loss, and
a full training run all run correctly and the loss decreases. I was not
able to produce a genuinely accurate trained model in the environment I
built this in, because it had no internet access to download MobileNetV2's
pretrained ImageNet weights, and training a backbone from random
initialization needs far more data and epochs than I could run there. Your
machine should have normal internet access, so `grid_detector_model.py`'s
default (pretrained weights) will actually apply — delete any
`grid_detector_model.pt` from this folder before your first real
training run, since it's a placeholder from that constrained test, not a
usable model.
