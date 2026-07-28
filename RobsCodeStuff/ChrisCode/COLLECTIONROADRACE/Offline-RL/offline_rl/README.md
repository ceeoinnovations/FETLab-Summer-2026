# Offline RL — isolated collect / train / deploy environment

Trains and runs an offline reinforcement learning policy (TD3+BC) for the
target-seeking task. Unlike the other option* projects, there's no
hardcoded control law here — a small neural network maps a compact visual
state directly to motor commands, learned from a fixed dataset of human
driving rather than a hand-written formula or live trial-and-error.

## Hardware & environment setup (do this first)

Same physical setup as the rest of this project family — only the
perception/control method differs here.

**1. Install Python dependencies:**
```
pip install torch torchvision opencv-python pillow numpy
```

**2. Check the hardware values in `config.py`:**
- `SERIAL` — the LEGO Bluetooth card serial for your motor + joystick controller.
- `SERIAL_COLOR_SENSOR` — a **separate** card for the color sensor. Must not
  share `SERIAL` — see the comment in `config.py` for why.
- `CAMERA` — the index Camo Studio's virtual webcam appears as.

**3. Before running anything:** Camo Studio must be running with your
phone connected, and the motor/controller hub + color sensor need to be
powered on and in range.

## Read this before collecting data

**This project needs human joystick driving specifically — not autonomous
driving from any other project, including this one's own trained actor.**

Offline RL can only learn "this action was better than that one" by
seeing the *same visual situation* handled with genuinely *different*
actions, with different outcomes. A deterministic controller (a hardcoded
formula, or another trained model driving itself) computes the same
action every time it sees the same input — so logging its own driving,
no matter how imperfect it looks, produces almost no usable contrast.
This was checked directly during development: autonomous data from two
different deterministic projects both measured at ~85-90% "within-
situation action variance relative to overall variance" (essentially no
local contrast) — genuinely not useful here. Real human driving measured
at ~35-40% on the same check, which is what actually let training work.

`check_data_diversity.py` runs that exact check on whatever you collect —
run it before `train_offline_rl.py`, not just after something looks wrong.

## Workflow

1. **`python collect_data.py`** — a human drives the car with the LEGO
   joystick. Collect several separate drives (press `N` between them).
   Vary your driving naturally — don't try to drive "perfectly" the same
   way every time; that natural variation is exactly what this method
   needs and a hardcoded controller can't provide.

2. **`python calibrate_color.py`** — tune `HSV_LOWER`/`HSV_UPPER` for your
   camera and lighting.

3. **`python generate_pseudo_labels.py data/images data/pseudo_labels.csv`**
   — runs the tuned color detector over every collected frame, recording
   `cx_norm, cy_norm, area_frac, visible` as the state. Spot-check a
   random sample against the source images before trusting it.

4. **`python check_data_diversity.py`** — read the verdict before
   proceeding. A ratio near 0.85-0.90 means don't bother training yet;
   go collect more varied human driving instead.

5. **`python train_offline_rl.py`** — trains `offline_rl_actor.py` via
   TD3+BC. Read the final action-std comparison it prints — if the
   learned std is much larger than the logged std, that's divergence
   (the critic overestimating out-of-distribution actions); much smaller
   means collapse (the actor settled near one "safe average" action).
   Neither is a usable checkpoint, regardless of how the loss curves looked.

6. **`python drive.py`** — runs autonomously using the trained actor.
   Falls back to a hardcoded search-and-spin reflex if the target isn't
   visible at all (the actor has little training experience with that
   case — see `drive.py`'s docstring).

## Files

- `collect_data.py`, `calibrate_color.py`, `config.py`, `lelib.py` —
  shared conventions with the rest of this project family
- `detect.py` — color-threshold detector only, no learned-detector option
  (see its docstring for why: perception must match training exactly)
- `generate_pseudo_labels.py` — turns collected images into the compact
  4-dim state
- `check_data_diversity.py` — the diagnostic described above; run it
  before training, not after something looks wrong
- `offline_rl_actor_model.py` — the actor (deployed) and critic
  (training-only) network definitions
- `train_offline_rl.py` — TD3+BC training with the stability recipe this
  project settled on after an earlier attempt diverged
- `drive.py` — deploys the trained actor; optionally exports its own
  driving as training data for *other* model types (see the honest
  caveat in `config.EXPORT_EXPERT_DATA`'s comment about why this doesn't
  usefully retrain *this* model specifically)

## What this will and won't do for you

Tested during development: on a dataset with genuine human-driving
diversity, this approach came out statistically tied with a plain
behavior-cloned model trained on the *same* data, in a closed-loop
evaluation using a learned dynamics model as a proxy simulator. On real
hardware, it's shown a genuine edge over some (but not all) other trained
approaches in this project family — better than the models trained by
imitating another model's own autonomous driving, not as good as a
project whose control law is hand-specified rather than learned. Don't
go in expecting this to obviously beat everything else; the honest
expectation, based on everything found so far, is "worth having as one
option among several," not "the best one."
