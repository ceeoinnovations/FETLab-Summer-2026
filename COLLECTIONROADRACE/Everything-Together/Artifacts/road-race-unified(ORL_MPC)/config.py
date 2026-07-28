"""
Unified config for the multi-model road-race switcher.

╔══════════════════════════════════════════════════════════════════════╗
║  EDIT THESE THREE VALUES FOR YOUR HARDWARE — nothing else in this    ║
║  file needs to change to swap robots/cards/cameras. Every mode (all  ║
║  8 models) reads SERIAL, SERIAL_COLOR_SENSOR, and CAMERA from here.  ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# Bluetooth card serial number printed on the LEGO card for the double motor.
SERIAL = 1227

# Dedicated card serial for the color sensor — DO NOT share SERIAL above.
# Pairing the color sensor to the same card as the double motor makes the
# hub bundle them into one drivebase+sensor profile, and it then applies
# its own built-in reaction (an automatic counterclockwise spin) whenever
# the sensor sees a near object — on top of, and fighting with, whatever
# motor commands the Python code sends. Using a separate physical card
# for the color sensor keeps the motors under our exclusive control.
SERIAL_COLOR_SENSOR = 2283

# Smartphone camera stream index/URL via Camo Studio (appears as a virtual
# webcam once Camo Studio is running and phone is connected). Use 0 for
# laptop webcam during testing without a phone.
CAMERA = 1

# ══════════════════════════════════════════════════════════════════════
# Everything below is shared tuning, preserved as-authored from each
# source project. You generally shouldn't need to touch these unless
# you're retuning a specific mode's behavior on real hardware.
# ══════════════════════════════════════════════════════════════════════

# Capture rate (Hz) the behavior-cloning data was collected at. Kept only
# for documentation purposes; not used at inference time.
CAPTURE_HZ = 10

# Motor speed range used to normalize/denormalize regression targets.
MOTOR_SPEED_MIN = -100
MOTOR_SPEED_MAX = 100

# Image size fed to every model (all built on MobileNetV2 @ 224x224).
IMG_SIZE = 224

# ── Motor deadzone ───────────────────────────────────────────────────────
# All modes can output small nonzero values even when nothing meaningful
# is intended — the double motor's closed-loop speed controller then
# chases that tiny target and produces visible jitter. Any commanded value
# with |value| <= MOTOR_DEADZONE is snapped to 0 before being sent.
MOTOR_DEADZONE = 1


def apply_deadzone(value, threshold=MOTOR_DEADZONE):
    """Return 0 if value is within the deadzone, else return value unchanged."""
    return 0 if abs(value) <= threshold else value


# ── Obstacle-avoidance interrupt (color sensor as a pseudo proximity sensor) ──
# Global for all 8 modes in this unified script — the color sensor is
# always connected, so every mode gets this safety reflex regardless of
# whether its original standalone project included it.
OBSTACLE_REFLECTION_THRESHOLD = 30  # raw 0-255 scale; tune to your test surface
AVOID_BACKUP_SPEED = -40
AVOID_BACKUP_TIME = 0.25
AVOID_TURN_DEGREES = 95
AVOID_DRIVE_SPEED = 40
AVOID_DRIVE_TIME = 1.0

# ══════════════════════════════════════════════════════════════════════
# Mode 1 — road-race-end-to-end
# End-to-end regression model. drive.py in the source project applied an
# extra 3x multiplier before the deadzone (unlike modes 5-8) — preserved
# exactly here.
# ══════════════════════════════════════════════════════════════════════
MODE1_MOTOR_MULTIPLIER = 3

# ══════════════════════════════════════════════════════════════════════
# Modes 5-8 — attempt1-4-road-race-expert-data
# Same end-to-end regression architecture as Mode 1, trained on data
# exported from Option 1's keypoint-regression model instead of a human
# demonstrator. No extra multiplier in the source drive.py.
# ══════════════════════════════════════════════════════════════════════
ATTEMPTS_MOTOR_MULTIPLIER = 1

# ══════════════════════════════════════════════════════════════════════
# Mode 2 — option1_keypoint_regression
# Hardcoded proportional controller driven by a learned keypoint detector
# (cx_norm, cy_norm, area_frac, visible).
# ══════════════════════════════════════════════════════════════════════
MODE2_STEER_GAIN = 10
MODE2_STEER_MAX = 60
MODE2_FORWARD_MAX_SPEED = 40
MODE2_FORWARD_SLOWDOWN_AREA = 0.08
MODE2_STOP_AREA_FRACTION = 0.22
MODE2_SEARCH_TURN_SPEED = 20
MODE2_SEARCH_REVERSE_DIRECTION_AFTER = 4.0  # seconds before flipping search direction
MODE2_DETECT_MEDIAN_WINDOW = 3
MODE2_DETECT_EMA_ALPHA = 0.3
MODE2_VISIBLE_THRESHOLD = 0.5

# ══════════════════════════════════════════════════════════════════════
# Mode 3 — option2_object_detection
# Same proportional-control shape as Mode 2, driven by a learned grid
# (YOLO-style) bounding-box detector instead. Tuned for slower approach.
# ══════════════════════════════════════════════════════════════════════
MODE3_STEER_GAIN = 10
MODE3_STEER_MAX = 60
MODE3_FORWARD_MAX_SPEED = 10
MODE3_FORWARD_SLOWDOWN_AREA = 0.08
MODE3_STOP_AREA_FRACTION = 0.22
MODE3_SEARCH_TURN_SPEED = 5
MODE3_SEARCH_REVERSE_DIRECTION_AFTER = 16.0
MODE3_DETECT_MEDIAN_WINDOW = 3
MODE3_DETECT_EMA_ALPHA = 0.3
MODE3_CONFIDENCE_THRESHOLD = 0.5

# ══════════════════════════════════════════════════════════════════════
# Mode 4 — option3_discrete_classification
# No continuous control at all — a trained classifier sorts each frame
# into one of CATEGORIES, majority-voted over the last few frames, and
# looked up directly in CATEGORY_MOTOR_COMMANDS.
# ══════════════════════════════════════════════════════════════════════
CATEGORIES = ["not_visible", "hard_left", "soft_left", "straight",
              "soft_right", "hard_right", "stop"]

SPEED_SCALE = 1
SPIN_FACTOR = 0.5
HARD_TURN_BALANCER = 2
SOFT_TURN_BALANCER = 1.25

CATEGORY_MOTOR_COMMANDS = {
    "not_visible": (round((20 * SPIN_FACTOR) * SPEED_SCALE), round((-20 * SPIN_FACTOR) * SPEED_SCALE)),
    "hard_left":   (round((5 * HARD_TURN_BALANCER) * SPEED_SCALE), round(40 * SPEED_SCALE)),
    "soft_left":   (round((20 * SOFT_TURN_BALANCER) * SPEED_SCALE), round(35 * SPEED_SCALE)),
    "straight":    (round(35 * SPEED_SCALE), round(35 * SPEED_SCALE)),
    "soft_right":  (round(35 * SPEED_SCALE), round((20 * SOFT_TURN_BALANCER) * SPEED_SCALE)),
    "hard_right":  (round(40 * SPEED_SCALE), round((5 * HARD_TURN_BALANCER) * SPEED_SCALE)),
    "stop":        (round(0 * SPEED_SCALE), round(0 * SPEED_SCALE)),
}

MODE4_CATEGORY_VOTE_WINDOW = 3

# ══════════════════════════════════════════════════════════════════════
# Mode 9 — offline RL (TD3+BC), trained on human joystick data
# Operates on a compact 4-dim state (cx_norm, cy_norm, area_frac,
# visible) from the classical color detector below, rather than an
# image directly — see models/offline_rl_actor_model.py for why.
# ══════════════════════════════════════════════════════════════════════

# HSV color range for the classical detector that supplies mode 9's live
# perception (color_detect.py) — same defaults used across this whole
# project's color-threshold detectors, and the same detector that
# generated the pseudo-labels this actor was trained on.
HSV_LOWER = (15, 80, 80)
HSV_UPPER = (40, 255, 255)
MIN_TARGET_AREA_FRACTION = 0.002

# The actor's output is already scaled and in the same -1..1-normalized
# action space as every other mode (see MOTOR_SPEED_MIN/MAX above) —
# denormalize_speed() handles converting it to real motor units, same as
# modes 1 and 5-8. The scale itself lives in models/offline_rl_actor_model.py,
# not here, since it must match what the checkpoint was trained with exactly.

# The training data mode 9 learned from had the target visible nearly
# 100% of the time (human demonstrations rarely lost track of it), so
# the actor has essentially no training signal for "target not visible."
# Rather than trust it outside its training distribution, mode 9 falls
# back to the same hardcoded search-and-spin reflex modes 2/3 use.
MODE9_SEARCH_TURN_SPEED = 20
MODE9_SEARCH_REVERSE_DIRECTION_AFTER = 4.0

# Smoothing for mode 9's live color-detector reading, same idea (and same
# _smooth() helper) as modes 2/3 — added after real-hardware testing showed
# unsmoothed per-frame noise causing a visible "slithering" oscillation on
# approach. NOTE: the actor was trained on RAW, unsmoothed pseudo-labels, so
# this introduces a small train/inference mismatch — smoothing the input the
# actor sees, when it never saw smoothed input during training. Expected to
# be a net win for the oscillation regardless; flagged here in case behavior
# shifts in some other subtle way too.
MODE9_DETECT_MEDIAN_WINDOW = 3
MODE9_DETECT_EMA_ALPHA = 0.3

# ══════════════════════════════════════════════════════════════════════
# Mode 0 — standalone model-predictive control (no wrapped policy)
# Every frame: sample many random candidate action sequences, roll each
# one through the learned dynamics model (models/mpc_dynamics_model.py),
# score the imagined outcomes, execute only the first action of the
# best-scoring sequence, then replan from scratch next frame. Nothing
# here is a trained policy in the usual sense — the "decision" is a
# search performed fresh every frame, not a fixed function of the input.
# ══════════════════════════════════════════════════════════════════════

# How many steps ahead each candidate sequence imagines. 5 specifically
# because the dynamics model was validated to modestly but genuinely beat
# a naive "assume nothing changes" baseline starting around 3 steps out,
# with the gap growing through at least 8 — shorter horizons don't let
# the model's real edge show up; much longer ones haven't been validated
# and the model's own accuracy keeps degrading the further out you go.
MODE0_HORIZON = 5

# How many random candidate sequences to sample and score per frame. All
# candidates for one frame are evaluated as a single batched forward pass
# per horizon step (HORIZON forward passes total, regardless of this
# number) — even 64-128 candidates is computationally trivial for a
# network this small, so this is set for planning quality, not because a
# larger number risked being too slow.
MODE0_NUM_CANDIDATES = 64

# Candidate actions are sampled uniformly in [-MODE0_ACTION_SAMPLE_RANGE,
# +MODE0_ACTION_SAMPLE_RANGE] on each wheel independently. This is the
# same ±0.2 scale the offline RL actor's own output was constrained to
# (see MODE9's ACTION_SCALE in offline_rl_actor_model.py) — keeping
# sampled candidates within the range the dynamics model actually saw
# during training avoids querying it on action magnitudes it has to
# extrapolate wildly beyond anything it learned from.
MODE0_ACTION_SAMPLE_RANGE = 0.2

# Reward used to score an imagined future state, identical to the one
# used to train the offline RL actor (mode 9) — "stay centered on the
# target." Kept simple and consistent rather than inventing a different
# objective for this mode.
MODE0_REWARD_NOT_VISIBLE = -1.0

# Replan every frame by default. If this turns out too slow on your
# hardware (unlikely given how small the dynamics model is, but this is
# the one part of this mode that's genuinely untested on real hardware —
# see the project notes on this), raise this to replan only every Nth
# frame instead, reusing the previous plan's action in between.
MODE0_REPLAN_EVERY_N_FRAMES = 1

# Search + detection-smoothing behavior when the target isn't visible —
# same reflex and same reasoning as mode 9 (the dynamics model has little
# training experience with "target not visible" either, so this doesn't
# trust the planner outside its training distribution).
MODE0_SEARCH_TURN_SPEED = 20
MODE0_SEARCH_REVERSE_DIRECTION_AFTER = 4.0
MODE0_DETECT_MEDIAN_WINDOW = 3
MODE0_DETECT_EMA_ALPHA = 0.3
