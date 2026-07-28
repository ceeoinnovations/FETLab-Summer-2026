# Option 1 — Keypoint regression

Perception method: a small trained CNN predicts a single (x, y) position
(plus a visibility flag) directly from the camera frame, skipping the
"find a bounding box" step entirely. Steering and speed are still decided
by the same hand-written proportional controller as the other two options.

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
   session markers. (Or reuse an existing dataset from the color-threshold
   project.)
2. `python calibrate_color.py` — tune HSV_LOWER/HSV_UPPER in `config.py`
   for your camera and target color. This is a bootstrapping step: the
   neural net's training labels come FROM this color detector, so it needs
   to already work reasonably well.
3. `python generate_pseudo_labels.py data/images data/pseudo_labels.csv`
   — runs the tuned color detector over every collected image and records
   its (cx_norm, area_frac, visible) output as training labels. Spot-check
   a random sample of the resulting CSV against the source images before
   trusting it.
4. `python train_detector.py data/images data/pseudo_labels.csv` — trains
   `detector_model.py`'s squeeze layer + last two backbone blocks against
   those pseudo-labels. Produces `detector_model.pt`.
5. `python drive.py` — runs autonomously. `config.DETECTOR_BACKEND` is set
   to `"neural_net"` by default in this project, so this uses the trained
   model, not the raw color threshold.

## Files

- `detector_model.py` — the perception network: a frozen MobileNetV2
  backbone whose spatial feature map is kept (never pooled) for a
  trainable squeeze + spatial-softmax path that predicts cx_norm/cy_norm,
  plus a separate ordinary pooled path for area_frac/visible (those two
  don't need precise location the way position does)
- `generate_pseudo_labels.py` — turns existing images into training labels
  using the color detector (always uses the color method, regardless of
  `DETECTOR_BACKEND`)
- `train_detector.py` — trains the model, with a masked loss so
  not-visible frames don't corrupt the position regression
- `detect.py` — `get_target()` dispatches to color or neural-net based on
  `config.DETECTOR_BACKEND`; `get_target_color()` always uses color
- `drive.py` — the control loop (unchanged from the color-threshold
  version — steering and speed math don't care which detector produced
  the centroid/size)
- `calibrate_color.py` — interactive HSV tuning tool, needed to bootstrap
  step 3 above even though the deployed car won't use color thresholding
- `collect_data.py`, `config.py`, `lelib.py` — data collection and
  hardware interface, shared with the other two options
