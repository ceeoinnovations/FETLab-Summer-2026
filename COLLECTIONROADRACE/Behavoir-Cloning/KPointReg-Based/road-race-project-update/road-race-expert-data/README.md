# road-race-expert-data

An end-to-end behavior-cloning project: one network takes a raw camera
frame and directly outputs `[left_speed, right_speed]` — no separate
perception step, no hand-written control law. This is deliberately the
"Raw Behavior Cloning" architecture from the very first version of this
whole project, revisited with one key change: **the training data now
comes from an already-working autonomous system driving itself, not a
human joystick.**

## Why train on "expert" data instead of human demonstrations

A human's joystick input is noisy, inconsistent frame to frame, and has
reaction-time lag — all of which land directly in the training labels.
`option1_keypoint_regression`'s `drive.py` (perception net + hand-written
proportional controller) is deterministic and continuous: for a given
scene it always produces the same, smoothly-scaled response. Recording
*its* behavior as training data gives this end-to-end model a cleaner,
more consistent teacher to imitate.

## Workflow

1. **Collect data by running the "expert," not this project.** In
   `option1_keypoint_regression/config.py`, set `EXPORT_EXPERT_DATA = True`,
   then run its `drive.py` as normal. It will drive autonomously as usual
   AND save each frame + the motor command it actually sent to
   `option1_keypoint_regression/expert_data/` (images + `labels.csv`, same
   format `collect_data.py` always used — filename, left_speed,
   right_speed, session_id). Run it across multiple sessions/drives for a
   variety of angles and distances, same advice as the original
   `collect_data.py`.
2. **Copy that exported folder here.** Copy
   `option1_keypoint_regression/expert_data/images` and
   `option1_keypoint_regression/expert_data/labels.csv` into this
   project's `data/` folder (`data/images/`, `data/labels.csv`) — no
   conversion needed, the format already matches what `train.py` expects.
3. **Train**: `python train.py` (or `python train.py <images_dir>
   <labels_csv>` to point at a different location). Produces
   `drive_model.pt` and a training curve plot.
4. **Drive**: `python drive.py` runs the trained model directly, frame in,
   motor command out.

## Files

- `model.py` — frozen MobileNetV2 backbone + regression head predicting
  `[left_speed, right_speed]` directly. Architecture unchanged from the
  original project, since the point of this experiment is specifically
  "does clean expert data fix raw behavior cloning, holding the
  architecture fixed" — not a redesign.
- `train.py` — trains on the imported data. Splits by whole session (not
  random per-frame) so validation holds out entire drives, and
  upweights turning frames during training so "go straight" doesn't
  drown out the rarer correction examples — same reasoning used
  throughout every other project in this series. Also unfreezes the
  last couple of backbone blocks (rather than the original's fully-frozen
  backbone), since a fully-frozen backbone was a real contributor to the
  original project's occasional wrong-direction turns.
- `drive.py` — loads `drive_model.pt` and drives directly from its
  output, with a motor deadzone applied (same small robustness fix used
  everywhere else in this series).
- `config.py`, `lelib.py` — hardware interface, unchanged.

There is no `collect_data.py` in this project — see the workflow above
for where data actually comes from.
