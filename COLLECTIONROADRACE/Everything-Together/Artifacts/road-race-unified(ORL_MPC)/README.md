# Road Race — Unified Model Switcher

A single script that connects the motors, color sensor, and camera once,
then lets you switch live between all 8 previously-trained driving
approaches by pressing a number key.

## Setup

1. Put this whole folder somewhere on the machine with Camo Studio, the
   LEGO Education Python SDK (`legoeducation`), PyTorch, and OpenCV
   already set up (same environment you used for the individual
   projects).
2. **Edit `config.py`** — the top of the file has exactly three values
   to set for your hardware:
   ```python
   SERIAL = 1227
   SERIAL_COLOR_SENSOR = 2283
   CAMERA = 1
   ```
   That's the only place you need to change anything. Every mode reads
   these same three constants.
3. Run it:
   ```
   python drive_unified.py
   ```

## Controls

| Key | Model |
|-----|-------|
| `1` | road-race-end-to-end (end-to-end regression, human demonstrations) |
| `2` | option1_keypoint_regression (hardcoded controller + learned keypoint detector) |
| `3` | option2_object_detection (hardcoded controller + learned grid/bbox detector) |
| `4` | option3_discrete_classification (learned classifier + motor-command lookup table) |
| `5` | attempt1-road-race-expert-data (end-to-end regression, trained on model #2's driving) |
| `6` | attempt2-road-race-expert-data (same, more training data) |
| `7` | attempt3-road-race-expert-data (same, more training data) |
| `8` | attempt4-road-race-expert-data (same, most training data) |
| `9` | offline_rl_human_data (TD3+BC offline RL, trained on human joystick data) |
| `0` | model_predictive_control (standalone MPC — plans via a learned dynamics model, no wrapped policy) |
| `q` | Stop motors and quit |

You can press a number key at any time — mid-drive — to switch models.
Each mode's internal smoothing/search/voting state resets on switch so
nothing stale carries over from whichever mode was active before.

## What's preserved from each source project

- **Motor scaling**: Mode 1 applies the same 3x multiplier to its
  regression output that the original `road-race-end-to-end/drive.py`
  did; modes 5–8 don't, matching `attempt*-road-race-expert-data/drive.py`.
  These are intentionally different — not a bug.
- **Per-mode tuning constants** (STEER_GAIN, FORWARD_MAX_SPEED,
  CATEGORY_MOTOR_COMMANDS, etc.) are unchanged from each source
  project's own `config.py`, just grouped under a `MODE2_`/`MODE3_`/etc.
  prefix in the unified `config.py` so they don't collide.
- **Model architectures** (`models/*.py`) are byte-for-byte the same
  network definitions used to train each checkpoint — verified here by
  loading every `.pt` file into its architecture with `strict=True` and
  confirming zero missing/unexpected keys.

## What changed from the original standalone projects

- **Obstacle avoidance is now global.** The color sensor is always
  connected in this unified script, so the same hardcoded avoidance
  reflex (back up, turn, drive around, turn back) that
  `road-race-end-to-end` and the three `option*` projects already had
  now also applies to modes 5–8, which didn't originally include it.
- **Single hardware config.** Previously, `option1_keypoint_regression`
  used a different `SERIAL`/`SERIAL_COLOR_SENSOR` pair than the other
  seven projects. This unified script uses one pair for all 8 modes —
  set it once in `config.py`.
- **All 8 models preload at startup** rather than loading on demand, so
  switching between them mid-drive is instant.

## Folder layout

```
road-race-unified/
├── drive_unified.py        # main script — run this
├── config.py               # EDIT HARDWARE CONSTANTS HERE
├── lelib.py                # shared LEGO Education hardware wrapper
├── models/
│   ├── end_to_end_model.py     # MobileNetV2 regression (modes 1, 5-8)
│   ├── detector_model.py       # keypoint regression detector (mode 2)
│   ├── grid_detector_model.py  # YOLO-style grid detector (mode 3)
│   └── classifier_model.py     # 7-category classifier (mode 4)
└── weights/
    ├── 1_road_race_end_to_end.pt
    ├── 2_option1_keypoint_regression.pt
    ├── 3_option2_object_detection.pt
    ├── 4_option3_discrete_classification.pt
    ├── 5_attempt1.pt
    ├── 6_attempt2.pt
    ├── 7_attempt3.pt
    └── 8_attempt4.pt
```

## Known open item

Mode 2 (option1_keypoint_regression) has a previously-identified
`STEER_GAIN` tuning issue causing overcorrection/oscillation on
hardware. No code change is needed — it's a constant-tuning task
(`MODE2_STEER_GAIN` / `MODE2_STEER_MAX` in `config.py`).


## Known Problems

Mode 9: Slithering problem. Attempted to fix by adding a smoothing function but the smoothing function actually got worse. Meaning that it's likely that this doesn't come from noisy perception.
Mode 0: 