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
