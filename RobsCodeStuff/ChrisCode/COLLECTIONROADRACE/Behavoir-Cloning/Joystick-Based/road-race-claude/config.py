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


def apply_deadzone(value, threshold=MOTOR_DEADZONE):
    """Return 0 if value is within the deadzone, else return value unchanged."""
    return 0 if abs(value) <= threshold else value
