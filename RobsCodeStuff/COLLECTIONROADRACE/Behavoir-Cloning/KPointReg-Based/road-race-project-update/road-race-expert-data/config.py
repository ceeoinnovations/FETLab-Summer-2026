# ── Change these values before running anything ──────────────────────────

# Bluetooth card serial number printed on the LEGO card. The same card
# can pair multiple devices (motor + controller), so one serial covers both.
SERIAL = 1227

# Smartphone camera stream index/URL via Camo Studio (appears as a
# virtual webcam once Camo Studio is running and phone is connected).
# Use 0 for laptop webcam during testing without a phone.
CAMERA = 1

# Capture rate (Hz) this project's data was collected at. There's no
# collect_data.py here anymore — this project trains purely on data
# EXPORTED from elsewhere (see train.py's docstring). This constant is
# kept only because it documents the frame spacing that data was
# captured at, which matters if you ever want to reason about motion
# between consecutive frames.
CAPTURE_HZ = 10

# Motor speed range used to normalize/denormalize regression targets.
# LEGO motor speeds are typically -100..100; adjust if your hub uses
# a different range.
MOTOR_SPEED_MIN = -100
MOTOR_SPEED_MAX = 100

# Image size fed to the model (matches MobileNetV2 input).
IMG_SIZE = 224

# ── Motor deadzone ─────────────────────────────────────────────────────────
# The trained model can output small nonzero values (a point or two) even
# when nothing meaningful is intended — the double motor's closed-loop
# speed controller then chases that tiny target and produces visible
# jitter. Any commanded value with |value| <= MOTOR_DEADZONE is snapped to
# 0 before being sent to the motors.
MOTOR_DEADZONE = 1  # same units as MOTOR_SPEED_MIN/MAX (-100..100)


def apply_deadzone(value, threshold=MOTOR_DEADZONE):
    """Return 0 if value is within the deadzone, else return value unchanged."""
    return 0 if abs(value) <= threshold else value

