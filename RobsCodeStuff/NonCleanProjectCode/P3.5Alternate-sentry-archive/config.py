# ── Change these values before running anything ──────────────────────────

# Bluetooth card serial number printed on the LEGO card. The same card
# can pair multiple devices (motor + controller), so one serial covers both.
SERIAL = 1227

# Smartphone camera stream index/URL via Camo Studio (appears as a
# virtual webcam once Camo Studio is running and phone is connected).
# Use 0 for laptop webcam during testing without a phone.
CAMERA = 1

# Capture rate (Hz) during the "motor is moving" phase of each iteration.
CAPTURE_HZ = 10

# Motor speed range used to normalize/denormalize regression targets and
# to clamp commanded speeds. Sign is direction: positive = clockwise,
# negative = counterclockwise (matches singleMotor.run()'s convention).
MOTOR_SPEED_MIN = -25
MOTOR_SPEED_MAX = 25

# Constant speed (magnitude) the motor runs at while approaching the
# target, before the force gauge detects contact.
APPROACH_SPEED = 10

# Force-gauge (right joystick) reading, in percent (0-100), above which
# we consider the shaft "in contact" with the target and start
# decelerating. Set a bit above the joystick's resting noise/deadzone.
CONTACT_THRESHOLD = 4

# Deceleration once contact is detected:
#   decel_rate = MIN_DECEL_RATE + DECEL_GAIN * force_reading   (units/sec)
# MIN_DECEL_RATE guarantees the motor always stops quickly even at low
# force; DECEL_GAIN makes harder contact stop it even faster on top of
# that floor. Both combine linearly with the force reading.
MIN_DECEL_RATE = 250.0
DECEL_GAIN = 6.0

# Below this speed magnitude we call the motor "stopped" and end the
# iteration (avoids an infinite asymptotic crawl toward exactly 0).
STOP_SPEED_EPSILON = 2.0

# Repositioning: after each trial, the motor drives back the opposite
# direction for a random duration to land on a new random start position
# before the next trial. There's no absolute position feedback on this
# rig, so a random dwell time at a fixed speed is what actually produces
# the "random position" - adjust the range to match your mount's travel.
REPOSITION_SPEED = 10
REPOSITION_MIN_SEC = 1.0
REPOSITION_MAX_SEC = 4.0

# Image size fed to the model (matches MobileNetV2 input, consistent with
# the driving-car project).
IMG_SIZE = 224

# Deployment-only: if the model's predicted speed magnitude falls below
# this, just call motor.stop() instead of commanding a tiny PWM value -
# avoids buzzing/jitter right around the centered position.
RUN_DEADBAND = 3
