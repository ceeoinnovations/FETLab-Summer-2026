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
# rather than guessing new values by hand.
HSV_LOWER = (15, 80, 80)
HSV_UPPER = (40, 255, 255)

# Ignore color blobs smaller than this fraction of the frame area — filters
# out small false-positive specks that aren't really the target.
MIN_TARGET_AREA_FRACTION = 0.002

# Which detector implementation get_target() uses — "color" (classical CV,
# no training, the safe fallback) or "neural_net" (this project's actual
# method — you must run generate_pseudo_labels.py then train_detector.py
# first, or this will fail to find a model file). See detect.py's
# docstring. Note: generate_pseudo_labels.py and calibrate_color.py always
# use the color method regardless of this setting — they exist to
# bootstrap/tune it, so they'd break if redirected to a not-yet-trained
# neural net.
DETECTOR_BACKEND = "neural_net"
PERCEPTION_MODEL_PATH = "detector_model.pt"

# Steering: turn amount (motor-speed units) per unit of normalized
# horizontal centroid error (-1 = target at left edge, +1 = right edge).
STEER_GAIN = 9
STEER_MAX = 60  # clamp so a large error can't command an extreme turn

# Forward speed vs. how close the target looks (its area as a fraction of
# the frame — bigger = closer). Drive at FORWARD_MAX_SPEED until the target
# starts filling FORWARD_SLOWDOWN_AREA of the frame, then linearly ease off
# to a full stop by STOP_AREA_FRACTION. This is a hard-specified stopping
# rule instead of something a network had to infer from noisy, latency-
# affected human demonstrations — tune the two area numbers by watching
# calibrate_color.py's area_frac reading at the distance you want to stop.
FORWARD_MAX_SPEED = 40
FORWARD_SLOWDOWN_AREA = 0.08
STOP_AREA_FRACTION = 0.33

# Search behavior when the target isn't visible at all: spin in place,
# favoring whichever direction it was last seen drifting toward (closer to
# where it probably still is), reversing direction periodically in case
# that guess was wrong.
SEARCH_TURN_SPEED = 20
SEARCH_REVERSE_DIRECTION_AFTER = 4.0  # seconds before flipping search direction

# Smoothing for the detected centroid/area, same idea as the joystick
# smoothing above — per-frame contour detection jitters a little even when
# the target hasn't moved (JPEG noise, lighting flicker), which would
# otherwise make the car's steering/speed twitch frame to frame.
DETECT_MEDIAN_WINDOW = 3
DETECT_EMA_ALPHA = 0.3

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


# ── Expert data export (optional) ───────────────────────────────────────────
# When True, drive.py ALSO saves each captured frame plus the motor command
# it actually computed and sent that frame, in the exact same (images/ +
# labels.csv) format collect_data.py already produces — so a separate
# end-to-end project (e.g. road-race-expert-data) can train directly on
# this autonomous run's own behavior, with no human joystick involved.
# This project's perception+control loop becomes the "expert" being
# imitated, instead of a human driving by hand.
#
# Capturing happens at EXPERT_CAPTURE_HZ, independent of the ~30Hz control
# loop, to avoid saving many near-duplicate frames — matching the original
# collect_data.py's capture rate keeps the exported dataset's frame-to-frame
# motion characteristics consistent with what a from-scratch human-driven
# dataset would have looked like.
EXPORT_EXPERT_DATA = True
EXPERT_DATA_DIR = "expert_data"   # created next to drive.py if it doesn't exist
EXPERT_CAPTURE_HZ = 10           # matches collect_data.py's CAPTURE_HZ

