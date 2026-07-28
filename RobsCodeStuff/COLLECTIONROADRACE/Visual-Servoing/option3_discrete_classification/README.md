# Option 3 — Discrete classification

Perception method: a trained classifier sorts each camera frame into one
of a small fixed list of categories (`config.CATEGORIES` — hard_left,
soft_left, straight, soft_right, hard_right, stop, not_visible) instead
of predicting any kind of position. There is no proportional controller
in this project at all — each category maps directly to a fixed
(left_speed, right_speed) pair via `config.CATEGORY_MOTOR_COMMANDS`, a
plain lookup table you tune by hand on real hardware.

This is the most structurally different of the three options: Options 1
and 2 both still hand off to continuous steering/speed math after
perception; this one's "control" step is just a dictionary lookup.

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
  color detector (used to generate training labels) can't distinguish
  your target from anything else in the same HSV range.
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
   Bootstrapping step, same reasoning as the other two options: category
   labels are auto-generated FROM this color detector's measurements.
3. `python generate_pseudo_classes.py data/images data/pseudo_classes.csv`
   — runs the tuned color detector over every image, then buckets its
   (cx_norm, area_frac) output into one of `config.CATEGORIES` using the
   `CX_HARD_TURN_THRESHOLD` / `CX_SOFT_TURN_THRESHOLD` / `STOP_AREA_FRACTION`
   boundaries in `config.py`. Prints a category count summary — if any
   category has very few or zero examples, collect more varied data
   (steeper approach angles in particular) before training.
4. `python train_classifier.py data/images data/pseudo_classes.csv` —
   trains `classifier_model.py` with cross-entropy loss and class-balanced
   sampling. Produces `classifier_model.pt`.
5. `python drive.py` — runs autonomously: classify each frame, majority-
   vote over the last few predictions, look up the fixed motor command
   for that category.

## Files

- `classifier_model.py` — MobileNetV2 backbone with ordinary global
  average pooling (no spatial-preserving trick — this option doesn't
  need one, since it never predicts a position) + a classification head
- `generate_pseudo_classes.py` — buckets the color detector's continuous
  output into discrete categories (always uses the color method directly)
- `train_classifier.py` — cross-entropy training with class-balanced
  sampling, since "straight"/"stop" are naturally far more common than
  "hard_left"/"hard_right" in most collected data
- `detect.py` — trimmed down from the other two options: just
  `get_target_color()` for bootstrapping, no dispatcher, since this
  project's `drive.py` never calls it at runtime
- `drive.py` — classifier → majority vote → fixed lookup table. No
  centroid, no bounding box, no proportional control anywhere in this file
- `calibrate_color.py`, `collect_data.py`, `config.py`, `lelib.py` —
  shared with the other two options

## A note on training this yourself

I validated this pipeline end-to-end against real project data — bucket
boundaries, class-balanced sampling, and a full training run all run
correctly. As with the other two options, I was not able to produce a
genuinely useful trained model in the environment I built this in (no
internet access to download pretrained ImageNet weights, and training a
backbone from random initialization needs far more data/epochs than I
could run there — the test model collapsed to predicting "not_visible"
for every single image, regardless of what was actually in frame). Your
machine should have normal internet access, so `classifier_model.py`'s
default (pretrained weights) will actually apply — delete any
`classifier_model.pt` from this folder before your first real training
run.

## A design choice worth knowing about

`CATEGORY_MOTOR_COMMANDS` bakes speed and steering into one lookup per
category (e.g. `"hard_left": (5, 40)`) rather than having separate
steering-only and speed-only categories. This keeps the category list
short, but means the classifier is implicitly doing a coarser version of
both perception AND judgment at once — "hard_left" isn't a pure geometric
fact like a centroid position, it already bakes in a decision about what
to do. If you want a cleaner separation, you could split this into two
independent classifiers (one for steering direction, one for stop/go)
that each feed their own part of the motor command — more moving parts,
but a more faithful reproduction of the perception/control split the
other two options have.
