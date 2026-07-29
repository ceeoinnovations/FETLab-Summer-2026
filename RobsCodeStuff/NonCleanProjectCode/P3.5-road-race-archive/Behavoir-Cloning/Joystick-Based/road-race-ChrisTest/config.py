# ── Change these values before running anything ──────────────────────────

# Bluetooth card serial number printed on the LEGO card. The same card
# can pair multiple devices (motor + controller), so one serial covers both.
SERIAL = 1227

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
