# ── Change these values before running anything ──────────────────────────

# Bluetooth card serial number printed on the LEGO card. The same card
# can pair multiple devices (motor + controller), so one serial covers both.
SERIAL = 1227

# Dedicated card serial for the color sensor — DO NOT share SERIAL above.
# Pairing the color sensor to the same card as the double motor makes the
# hub bundle them into one drivebase+sensor profile, and it then applies
# its own built-in reaction (an automatic counterclockwise spin) whenever
# the sensor sees a near object — on top of, and fighting with, whatever
# motor commands the Python code sends. Using a separate physical card
# for the color sensor keeps the motors under our exclusive control.
SERIAL_COLOR_SENSOR = 2283  # print/use a second LEGO card for this

# Smartphone camera stream index/URL via Camo Studio (appears as a
# virtual webcam once Camo Studio is running and phone is connected).
# Use 0 for laptop webcam during testing without a phone.
CAMERA = 1

# Capture rate (Hz) for data collection.
CAPTURE_HZ = 10

# Motor speed range used to normalize/denormalize actions. LEGO motor
# speeds are typically -100..100; adjust if your hub uses a different range.
MOTOR_SPEED_MIN = -100
MOTOR_SPEED_MAX = 100

# ── Why this project exists, and what NOT to feed it ────────────────────────
# This project trains an offline reinforcement learning policy (TD3+BC) —
# NOT a behavior-cloned model. That distinction matters for what data is
# actually useful here, and it's worth being upfront about since it's easy
# to assume "more driving data" is automatically good data for this purpose.
#
# Offline RL needs the SAME visual situation to have been handled with a
# genuinely DIFFERENT action more than once in the data, with different
# outcomes — that contrast is what lets it learn "this action was better
# than that one here," rather than just copying whatever was recorded.
#
# Autonomous data from a hardcoded or learned deterministic controller
# (e.g. a proportional visual-servoing project, or another trained
# end-to-end model driving itself) does NOT provide this. A deterministic
# policy computes the same action every time it sees the same input, so
# logging its own driving — no matter how imperfect the driving looks —
# produces almost no same-state-different-action contrast. This was
# checked directly during development: two different autonomous projects'
# logged driving both came back with within-situation action variance
# at ~85-90% of the dataset's overall variance (i.e. almost no local
# contrast at all) — genuinely not useful here, regardless of how "messy"
# or "different" the driving looked.
#
# Real human joystick driving (collect_data.py) is what worked — natural
# human variation gives real contrast (the same check came back at ~35-40%
# instead of ~85-90% on genuine human demonstrations). ALWAYS use
# collect_data.py, driven by a human, for this project. Do not point
# train_offline_rl.py at another project's autonomous drive.py output,
# even from this same folder's own trained actor.
#
# Run check_data_diversity.py after collecting data and before training —
# it runs the same diagnostic and will tell you directly whether what you
# collected has enough contrast to be worth training on.

# ── Perception: classical HSV color detector ────────────────────────────────
# This project always uses the color detector (detect.py) for perception,
# both to generate training pseudo-labels and for live driving — no learned
# detector option here. Consistency between training and inference
# perception matters more than perception quality for this project
# specifically: the actor learns a mapping FROM whatever detector's output,
# so using a different (even if "better") detector at inference than the
# one that generated training labels would feed it inputs it never learned
# to interpret.
#
# Defaults below were tuned for a bright yellow LEGO plate against plain
# gray walls/table. Camera, lighting, and target color all vary station to
# station — run calibrate_color.py rather than guessing new values by hand.
HSV_LOWER = (15, 80, 80)
HSV_UPPER = (40, 255, 255)
MIN_TARGET_AREA_FRACTION = 0.002

# ── The compact state the actor operates on ──────────────────────────────────
# (cx_norm, cy_norm, area_frac, visible) — 4 numbers, not an image. An
# earlier attempt at this project tried a much richer state (a full 1280-dim
# MobileNetV2 feature vector) and it overfit badly: with the amount of data
# a single data-collection session realistically produces, a dynamics/value
# model over that many dimensions ended up WORSE than a trivial "assume
# nothing changes" baseline. Collapsing to this 4-dim, task-relevant state
# is what made training stable at all — this isn't a simplification made
# for speed, it's a fix for a real failure mode. Don't be tempted to swap
# in a learned visual backbone here without re-validating that decision.
STATE_DIM = 4
ACTION_DIM = 2

# ── Reward ───────────────────────────────────────────────────────────────────
# reward = 1 - |cx_norm| when the target is visible ("stay centered"),
# REWARD_NOT_VISIBLE when it isn't. Simple and coarse on purpose — every
# more elaborate reward considered during development (rewarding area_frac
# increasing, smoothness penalties) added complexity without a validated
# benefit, and this one is easy to reason about and to recompute by hand
# from pseudo_labels.csv while debugging.
REWARD_NOT_VISIBLE = -1.0

# ── TD3+BC (offline RL) training hyperparameters ─────────────────────────────
# TD3+BC = TD3 (twin-critic actor-critic) + a behavior-cloning regularization
# term that keeps the learned policy from straying too far from the actions
# actually in the dataset. That regularization is NOT optional tuning — an
# early attempt with ALPHA=1.5 (less BC weight, more pure Q-maximization)
# diverged: critic Q-values grew unbounded and the actor collapsed to
# near-constant, extreme-looking actions far outside anything in the data.
# There is no live environment to catch this the way online RL would (see
# this project's earlier discussion of what makes this "offline" in the
# first place) — nothing corrects an overconfident critic except staying
# close to the data, which is exactly what a low ALPHA enforces.
# ALPHA=0.4 is the value that actually trained stably (converging BC error,
# an action distribution matching the real data's spread) across every
# dataset this was tried on. Raise it only with real caution, and always
# check the printed action-std comparison at the end of training against
# both failure directions actually seen during development: much LARGER
# than the logged std means divergence (the critic overestimating
# out-of-distribution actions); much SMALLER means collapse (the actor
# settling near one "safe average" action regardless of state, seen when
# a training run is too small/uninformative for the actor to learn
# anything more specific). Neither checkpoint should be deployed.
GAMMA = 0.99                  # discount factor
TAU = 0.005                   # target network soft-update rate
POLICY_NOISE = 0.02            # smoothing noise added to the target action
NOISE_CLIP = 0.05
ALPHA = 0.4                    # BC-regularization weight -- see note above
HIDDEN_SIZE = 64                # width of both the actor and critic MLPs
CRITIC_LR = 1e-4
ACTOR_LR = 1e-4
CRITIC_WEIGHT_DECAY = 1e-3
BATCH_SIZE = 128
TRAIN_STEPS = 5000
EVAL_EVERY = 400               # steps between validation checks / checkpointing
VAL_FRACTION = 0.15
TRAIN_TEST_SPLIT_SEED = 0

# The actor's raw tanh output is multiplied by this before being treated as
# a normalized (-1..1) action and denormalized to real motor units — matches
# the actual range of actions in typical joystick data (MOTOR_DIVIDE_BY-style
# scaling means raw labels rarely exceed about ±20 out of the ±100 the
# motors are physically capable of). This is baked into
# offline_rl_actor_model.py's forward() itself (not applied by callers),
# since it must exactly match what a trained checkpoint's weights assume —
# changing this number after training invalidates any existing checkpoint.
ACTION_SCALE = 0.2

# ── Search behavior when the target isn't visible ────────────────────────────
# The actor is trained almost entirely on frames where the target IS visible
# (that's what human demonstrations mostly look like) — it has little to no
# experience with "target not visible" and shouldn't be trusted to improvise
# there. drive.py falls back to this hardcoded spin-and-search reflex
# instead of trusting the actor outside its training distribution, same
# idea as the other option* projects' search behavior.
SEARCH_TURN_SPEED = 20
SEARCH_REVERSE_DIRECTION_AFTER = 4.0  # seconds before flipping search direction

# ── Live-perception smoothing ────────────────────────────────────────────────
# Median filter + EMA on the detector's cx_norm/cy_norm/area_frac before the
# actor sees them. Added after real-hardware testing showed unsmoothed
# per-frame detector noise (JPEG artifacts, lighting flicker) causing a
# visible "slithering" oscillation on approach — the actor has no memory of
# its own and no explicit smoothness constraint, so tiny frame-to-frame
# wobbles in perceived position translated into a jumpier response than a
# hand-written proportional controller (which only ever saw smoothed input
# to begin with) would ever produce.
#
# NOTE: the actor was trained on RAW, unsmoothed pseudo-labels — this
# introduces a small train/inference mismatch (the actor never technically
# saw smoothed input during training). Net effect on real hardware was
# positive (oscillation fixed, no other issues observed), but if you notice
# a different subtle behavior change, this mismatch is the first thing to
# suspect — try lowering DETECT_EMA_ALPHA before assuming something else
# is wrong.
DETECT_MEDIAN_WINDOW = 3
DETECT_EMA_ALPHA = 0.3

# Image size — not actually used for model input (the actor takes the
# 4-dim state, not an image), but detect.py's color threshold operates
# on the frame at its native camera resolution regardless of this value.
# Kept for parity with the other option* projects' config.py.
IMG_SIZE = 224

# ── Obstacle-avoidance interrupt (color sensor used as a pseudo proximity sensor) ──
OBSTACLE_REFLECTION_THRESHOLD = 30  # raw 0-255 scale; tune to your test surface
AVOID_BACKUP_SPEED = -40
AVOID_BACKUP_TIME = 0.25
AVOID_TURN_DEGREES = 95
AVOID_DRIVE_SPEED = 40
AVOID_DRIVE_TIME = 1.0

# ── Motor deadzone ───────────────────────────────────────────────────────────
MOTOR_DEADZONE = 1


def apply_deadzone(value, threshold=MOTOR_DEADZONE):
    """Return 0 if value is within the deadzone, else return value unchanged."""
    return 0 if abs(value) <= threshold else value


# Raw joystick (-100..100) is divided by this before being saved as the
# training label in collect_data.py, giving labels roughly in -17..17 —
# matches how every other project in this family collects joystick data,
# so datasets are comparable in scale. Do NOT add any other multiplier in
# drive.py — both files must send apply_deadzone(speed) with no extra
# scaling, so a given label value means the exact same physical motor
# command whether a human demonstrated it or the trained actor predicted it.
MOTOR_DIVIDE_BY = 6

# ── Joystick smoothing (collect_data.py) ─────────────────────────────────────
SMOOTH_MEDIAN_WINDOW = 3
SMOOTH_EMA_ALPHA = 0.15

# ── Expert data export (optional, in drive.py) ───────────────────────────────
# Same mechanism as the other option* projects: when True, drive.py ALSO
# saves each frame + the motor command it actually sent, in the same
# (images/ + labels.csv) format collect_data.py produces.
#
# HONEST CAVEAT specific to THIS project: because the trained actor is a
# deterministic function of its input (no exploration noise at inference),
# logging its own autonomous driving is subject to the exact same
# same-state-different-action problem described above — it's unlikely to
# usefully improve THIS offline RL model if fed back into train_offline_rl.py.
# It's still useful as general coverage data for other model types in this
# family (e.g. behavior-cloned end-to-end models), which is why the feature
# is kept here rather than removed — just don't expect it to bootstrap a
# better version of this specific model.
EXPORT_EXPERT_DATA = True
EXPERT_DATA_DIR = "expert_data"
EXPERT_CAPTURE_HZ = 10
