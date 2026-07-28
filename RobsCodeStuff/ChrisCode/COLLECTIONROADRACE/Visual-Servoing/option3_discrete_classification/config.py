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

# Capture rate (Hz) for behavior-cloning data collection.
# 10 Hz balances BLE polling latency (~30-50ms per read) against
# capturing meaningful steering corrections.
CAPTURE_HZ = 10

# Motor speed range used to normalize/denormalize regression targets.
# LEGO motor speeds are typically -100..100; adjust if your hub uses
# a different range.
MOTOR_SPEED_MIN = -100
MOTOR_SPEED_MAX = 100

# ── Vision-based target-seeking (bounding box + centroid control) ──────────
# This replaces the learned end-to-end steering model for the core driving
# task: detect.py finds the target's bounding box via color thresholding,
# and drive.py steers/slows down using its centroid + size directly (a
# simple proportional controller), instead of a CNN trying to imitate
# human joystick demonstrations. See detect.py for why.

# HSV color range that matches the target. Defaults below were tuned for a
# bright yellow LEGO plate against plain gray walls/table. Camera, lighting,
# and target color all vary station to station — run calibrate_color.py
# rather than guessing new values by hand. Only used to bootstrap category
# labels (generate_pseudo_classes.py) — the deployed car never runs this
# directly, since drive.py uses the trained classifier instead.
HSV_LOWER = (15, 80, 80)
HSV_UPPER = (40, 255, 255)

# Ignore color blobs smaller than this fraction of the frame area — filters
# out small false-positive specks that aren't really the target.
MIN_TARGET_AREA_FRACTION = 0.002

# ── Discrete categories ─────────────────────────────────────────────────
# The one list that defines this whole approach. Order matters only in
# that it fixes which index the network's output corresponds to — the
# actual class ordering doesn't matter otherwise, but changing this list
# means retraining, and changing CATEGORY_MOTOR_COMMANDS below without
# retraining is fine (that lookup is pure post-hoc code, not learned).
CATEGORIES = ["not_visible", "hard_left", "soft_left", "straight",
              "soft_right", "hard_right", "stop"]

# Boundaries used ONLY when auto-bucketing pseudo-labels into categories
# (generate_pseudo_classes.py) — the trained classifier itself has no
# notion of these numbers afterward, it just learned to associate images
# with category names directly.
CX_HARD_TURN_THRESHOLD = 0.65   # |cx_norm| beyond this -> hard_left / hard_right
CX_SOFT_TURN_THRESHOLD = 0.15  # |cx_norm| beyond this (but under HARD) -> soft_left / soft_right
STOP_AREA_FRACTION = 0.22      # area_frac at/above this -> stop, regardless of cx_norm

# What the car actually does for each category — a plain lookup table,
# not a formula. Tune these by testing on real hardware, the same way
# STEER_GAIN/FORWARD_MAX_SPEED needed real tuning in the other two
# options; there's no principled way to derive them, only trial and error.
SPEED_SCALE = 1 # Global speed scaling
SPIN_FACTOR = 0.5 #Change how quickly it spins when it says "not_visible"
HARD_TURN_BALANCER = 2 # Balances hard turns. Prevents over correction. Higher the number, the faster the lagging wheel goes.
# ^^ Should be 0 < x < 8
SOFT_TURN_BALANCER = 1.25 # Balances soft turns. Prevents oscillation. Higehr the number, the faster the lagging wheel goes.
# ^^ Should be 0 < x < 1.75
CATEGORY_MOTOR_COMMANDS = {
    "not_visible": (round((20*SPIN_FACTOR)*SPEED_SCALE), round((-20*SPIN_FACTOR)*SPEED_SCALE)),   # spin in place to search
    "hard_left":   (round((5*HARD_TURN_BALANCER)*SPEED_SCALE), round(40*SPEED_SCALE)),
    "soft_left":   (round((20*SOFT_TURN_BALANCER)*SPEED_SCALE), round(35*SPEED_SCALE)),
    "straight":    (round(35*SPEED_SCALE), round(35*SPEED_SCALE)),
    "soft_right":  (round(35*SPEED_SCALE), round((20*SOFT_TURN_BALANCER)*SPEED_SCALE)),
    "hard_right":  (round(40*SPEED_SCALE), round((5*HARD_TURN_BALANCER)*SPEED_SCALE)),
    "stop":        (round(0*SPEED_SCALE), round(0*SPEED_SCALE)),
}

PERCEPTION_MODEL_PATH = "classifier_model.pt"

# Smoothing for the predicted category: majority vote over the last N
# frames, so a single flickered misclassification doesn't cause a visible
# twitch in motor command. Numeric smoothing (median/EMA) doesn't apply to
# a category the way it did for cx_norm/area_frac in the other two
# options — there's no "average" of "hard_left" and "straight" — so this
# project uses a vote instead.
CATEGORY_VOTE_WINDOW = 3

# Image size fed to the model (matches MobileNetV2 input).
IMG_SIZE = 224

# ── Obstacle-avoidance interrupt (color sensor used as a pseudo proximity sensor) ──
# The color sensor's LED bounces light off whatever is in front of it; a nearby
# object reflects much more light back than the open floor/background ahead of
# the car, so a spike in `reflection()` (0-255) is a reliable "something is
# right in front of me" signal without needing a dedicated distance sensor.
# If your environment has a very bright floor, use saturation() instead — see
# the comment above the check in drive.py.
OBSTACLE_REFLECTION_THRESHOLD = 30  # raw 0-255 scale; tune to your test surface

# Hardcoded avoidance maneuver: turn away from the target, drive blind for a
# fixed duration to clear the obstacle, then turn back before resuming search.
AVOID_BACKUP_SPEED = -40
AVOID_BACKUP_TIME = 0.25
AVOID_TURN_DEGREES = 95     # degrees to turn left, then right, to go around
AVOID_DRIVE_SPEED = 40      # motor speed (0-100) while driving around the obstacle
AVOID_DRIVE_TIME = 1.0      # seconds to drive after the left turn

# ── Motor deadzone ─────────────────────────────────────────────────────────
# Both the joystick (collect_data.py) and the trained model (drive.py) can
# output small nonzero values (+1..-1 range observed) even when nothing is
# meaningfully "pushed" — the double motor's closed-loop speed controller
# then chases that tiny target and produces visible jitter. Any commanded
# value with |value| <= MOTOR_DEADZONE is snapped to 0 before being sent to
# the motors, so small noise no longer causes movement.
MOTOR_DEADZONE = 1  # same units as MOTOR_SPEED_MIN/MAX (-100..100)

# Raw joystick (-100..100) is divided by this before being saved as the
# training label, giving labels roughly in -17..17. If you want the car to
# move faster/slower overall, change this number — but do NOT add any other
# multiplier in collect_data.py or drive.py. Both files must send
# `apply_deadzone(speed)` to the motors with no extra scaling, so that a
# given label value means the exact same physical motor command whether a
# human demonstrated it or the trained model predicted it. (A previous
# version of this codebase divided by 10 in collect_data.py and multiplied
# by 3 in drive.py — a hidden 30x mismatch between demonstrated and deployed
# speed — which was a major cause of the car overshooting and pushing the
# target instead of stopping on it.)
MOTOR_DIVIDE_BY = 6

# ── Joystick smoothing ──────────────────────────────────────────────────────
# Analog joysticks (and small human thumb tremor) can produce single-frame
# spikes far from what you actually intended — the car "lurches" even though
# you meant a soft, gradual correction. This runs the raw joystick reading
# through a small median filter (rejects a single outlier sample) followed by
# light exponential smoothing, before it's used for anything — both the
# recorded label and what's mirrored to the motors live.
#
# The median filter does most of the work and costs very little lag (it just
# needs 2 of the last 3 samples to agree, so it "believes" a genuine change
# about 1 frame/~100ms after it happens) — a single-frame glitch never gets
# 2 votes, so it's fully rejected. The EMA term adds a bit more softness for
# residual hand tremor, but each 0.1 of SMOOTH_EMA_ALPHA costs roughly another
# frame of lag before a genuine full-speed or full-stop input takes effect.
# Keep SMOOTH_EMA_ALPHA low (or 0) if you notice stopping got less precise —
# that delay stacks on top of the car's own ~300ms stop/motor-coast lag.
SMOOTH_MEDIAN_WINDOW = 3   # odd number of recent samples to median-filter over
SMOOTH_EMA_ALPHA = 0.15    # 0 = median filter only, higher = smoother but laggier


def apply_deadzone(value, threshold=MOTOR_DEADZONE):
    """Return 0 if value is within the deadzone, else return value unchanged."""
    return 0 if abs(value) <= threshold else value
