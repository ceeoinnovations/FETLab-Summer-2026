import io
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi.responses import StreamingResponse
from PIL import ImageDraw

from arduino.app_utils import App, Logger
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.object_detection import ObjectDetection
from arduino.app_peripherals.camera import Camera
from arduino.app_utils.image import numpy_to_pil, draw_bounding_boxes

import legoeducation
import lelib as le
from legoeducation import BridgeBusyError, EXIT_BRIDGE_BUSY

import homography
import imaging
import tracking_cycle

logger = Logger("SpiderKicker")

# ── Rig configuration ────────────────────────────────────────────────────
# Bluetooth connection-card serial for each winch's LEGO Single Motor hub
# (printed on the motor's connection card). Order matches ANCHORS below -
# motors[i] winds cable i, anchored at ANCHORS[i]. Update to your hubs.
MOTOR_SERIALS = ["2287", "1126", "1392", "0002"]

# Anchor points (x', y', z') in mm that each cable runs from - e.g. the four
# top corners of the rig frame. Measure your physical setup and edit these.
ANCHORS = [
    (0.0, 0.0, 0.0),
    (977.9, 0.0, 0.0),
    (962.0, 939.8, 0.0),
    (0.0, 939.8, 0.0),
]

# Per-cable attachment offset (dx, dy, dz) in mm, same axis convention as
# ANCHORS/HOME_POSITION: how far cable i's actual attachment point on the
# kicker body sits from the single (x, y, z) reference point tracked
# everywhere else in this file (HOME_POSITION, current_position, move_to()'s
# argument). Order matches ANCHORS/MOTOR_SERIALS - offset i belongs to the
# cable anchored at ANCHORS[i]. Measure your physical setup and edit these;
# defaults to all zeros (no correction) so nothing breaks before you have.
#
# The x/y components correct for the redesigned gondola body, whose 4 cable
# attachment points were deliberately spread outward from center to reduce
# oscillation. The z component corrects a separate, pre-existing offset
# between the tracked reference point's height and the cables' true
# attachment height - it's the same physical height for all 4 cables, so
# enter the same dz in all 4 tuples.
ATTACHMENT_OFFSETS = [
    (-53.975, -31.75, -31.75),
    (53.975, -31.75, -31.75),
    (53.975, 31.75, -31.75),
    (-53.975, 31.75, -31.75),
]

# Camera's actual physical position (x, y, z) in mm when this script starts.
# Each motor's relative-position counter is reset to 0 here, so the hub -
# not this script - is the source of truth for how far it's turned since.
# Physically place the camera at this position *before* launching the app.
HOME_POSITION = (495.3, 476.25, 714.375)

# Diameter (mm) of the spool each cable winds around on its motor shaft.
# Degrees-of-rotation per mm of cable is derived from this - measure it
# accurately or the commanded lengths will be off from reality.
SPOOL_DIAMETER_MM = 37.2
MM_PER_DEGREE = math.pi * SPOOL_DIAMETER_MM / 360.0

# Sign that maps "increasing relative position" to "reeling in (shortening)"
# each cable. If a motor pays cable OUT as its relative position increases,
# flip that entry to -1.
REEL_IN_SIGN = [1] * 4

MOTOR_SPEED = 40  # percent, applied to every winch move

# Manual jog controls (web UI "wind" buttons): each click turns the motor by
# a fixed step at a slower, more controllable speed than a targeted move.
JOG_STEP_DEGREES = 45
JOG_SPEED = 25  # percent

# Web UI +/- buttons: each click nudges one axis of the current target
# position by this many mm, then re-issues a move to the updated position.
NUDGE_STEP_MM = 25.0
AXES = ("x", "y", "z")

# Kicker motor (web UI "Kick!" button): a standalone Single Motor, not part
# of the winch rig - just spins a fixed amount at full speed on demand.
KICKER_MOTOR_SERIAL = "1133"
KICKER_ROTATIONS = 1.5
KICKER_SPEED = 100  # percent, full speed

# ── Autonomous tracking configuration ────────────────────────────────────
# Class names the trained Edge Impulse model (see app.yaml) outputs.
PLATFORM_LABEL = "motor"
BALL_LABEL = "ball"
DETECTION_CONFIDENCE = 0.5

CAMERA_RESOLUTION = (640, 480)
# The deployed Edge Impulse model's own declared input is a 480x480 square
# (confirmed via the ei-obj-detection-runner container's startup log).
# 640x480 is the camera's least-lossy native resolution for that target -
# height (480) already matches exactly (0 loss vertically), only width needs
# trimming from 640 to 480 (center_crop() below), discarding just 25% of the
# frame - much gentler than the 66.7% loss a 320x320 target required. This
# also empirically measured better than padding (keeping 100% of content but
# not "zooming in" on these small objects) or feeding the mismatched aspect
# ratio uncropped and letting the runner resize it - see handoff notes:
# cropping effectively magnifies the ball/motor (already only a handful of
# pixels each), which seems to matter more for this detector than how much
# background survives. pad_to_square() is kept below as an easy fallback to
# toggle back to if cropping is ever the wrong call again (e.g. a retrained
# model with a different Resize mode).
MODEL_INPUT_SIZE = 480

# Manual exposure override: the camera's auto-exposure ("Aperture Priority
# Mode") washes out the image with glare under this rig's lighting, hurting
# detection accuracy. These are just the starting values applied at boot -
# tune live via the web UI's debug panel (no restart needed).
MANUAL_EXPOSURE_ENABLED = True
MANUAL_EXPOSURE_VALUE = 250  # exposure_time_absolute units (1-5000); starting
                             # guess of half the auto-mode default (156) -
                             # there's no universally "correct" value, re-tune
                             # by watching the live feed under real lighting.
EXPOSURE_MIN = 1
EXPOSURE_MAX = 800

# Floor for how often to capture/detect/publish a frame. nudge()/kick() calls
# already block for real motor-move time during autonomous driving, so this
# doesn't need to pace those - it only stops the loop from busy-spinning
# when capture+detect return faster than this. Lower = higher perceived fps
# on the video feed, at the cost of more CPU/network spent on detection.
LOOP_PERIOD = 0.05  # seconds

KICK_COOLDOWN_S = 3.0  # defensive-only guard now - a kick only ever fires once
                       # per fully-verified cycle, which already takes several
                       # real seconds, but this stays as a cheap safety net.
LOST_TARGET_TIMEOUT_S = 1.5  # discard stale accumulated ball samples / fail a
                              # stuck VERIFYING phase after this long without
                              # a fresh detection - see tracking_cycle.py.

# Where calibrate.py's saved homography lives - see calibrate.py/snapshot.py
# for the one-time (and re-run-after-camera-moves) calibration workflow.
CALIBRATION_PATH = Path(__file__).parent / "calibration.npz"

# How many ball detections to average (in world mm, not pixels) before
# committing to a target and driving there - see tracking_cycle.py.
SAMPLE_COUNT = 5

# Placeholder z-values (mm) - TUNE ON THE PHYSICAL RIG. Payload hangs BELOW
# the anchors (z=0), so SMALLER z = higher/lifted, LARGER z = lower/dropped.
TRAVEL_HEIGHT_MM = 810.0            # lifted height while translating - must clear obstructions
DROP_HEIGHT_MM = HOME_POSITION[2]-100   # starts at HOME_POSITION's known-safe z; will need adjusting

VERIFY_TOLERANCE_MM = 40.0  # how close the re-detected platform must land to the target to accept it
MAX_VERIFY_RETRIES = 0
VERIFY_CORRECTION_GAIN = 1.0

# How long to wait after a verified landing before kicking, so the
# cable-suspended platform's post-drop swing damps out first.
KICK_SETTLE_S = 1.5

# Fixed offset (mm, world axes) applied to the ball's target point, to
# account for the kicker mechanism not being at the platform's visual
# center. Tune by testing once the physical kicker mount is known.
KICKER_OFFSET_MM = (0.0, 0.0)


def cable_lengths(position):
    """L_i = distance from ANCHORS[i] to cable i's actual attachment point on
    the kicker body, which is offset from `position` by ATTACHMENT_OFFSETS[i]
    (dx, dy, dz) - not `position` itself.

    Simplifying assumption: the kicker body's orientation relative to world
    axes stays effectively constant as it translates, so each attachment
    point in world coordinates is just `position + ATTACHMENT_OFFSETS[i]`, a
    fixed per-cable vector added regardless of where `position` is. This is
    NOT a full cable-driven-parallel-robot pose solve - it doesn't track or
    correct for rotation of the body. That's fine as long as the wider/more
    stable attachment geometry keeps rotation negligible in practice. If the
    kicker ever starts visibly rotating during moves (not just settling at
    rest), this per-cable constant-offset model breaks down and the offsets
    would need to become functions of the body's actual orientation - that's
    a materially bigger change (tracking/estimating orientation, then
    rotating each offset vector into world frame before adding it here) and
    is out of scope for this fix.
    """
    x, y, z = position
    return [
        math.sqrt((ax - (x + dx)) ** 2 + (ay - (y + dy)) ** 2 + (az - (z + dz)) ** 2)
        for (ax, ay, az), (dx, dy, dz) in zip(ANCHORS, ATTACHMENT_OFFSETS)
    ]


def connect_motor(serial, label):
    motor = le.singleMotor()
    logger.info(f"Connecting to {label} (card_serial={serial})...")
    try:
        motor.connect(card_serial=serial)
    except BridgeBusyError as e:
        # Distinct exit code so a supervisor can tell "bridge stuck from a
        # previous run" apart from other crashes - see legoeducation.py.
        logger.error(f"{e} Exiting with code {EXIT_BRIDGE_BUSY}.")
        sys.exit(EXIT_BRIDGE_BUSY)
    if not motor.connected:
        raise ConnectionError(f"Could not connect to {label} - check its card serial.")
    return motor


# ── Connect all 4 winch motors, plus the kicker motor ────────────────────
motors = [connect_motor(serial, f"winch motor {i + 1}") for i, serial in enumerate(MOTOR_SERIALS)]
for motor in motors:
    motor.motor_reset_relative_position()  # zero the hub's own counter at HOME_POSITION
logger.info("All 4 winch motors connected and zeroed at HOME_POSITION.")

kicker_motor = connect_motor(KICKER_MOTOR_SERIAL, "kicker motor")
logger.info("Kicker motor connected.")

camera = Camera(resolution=CAMERA_RESOLUTION)
camera.start()

_exposure_lock = threading.Lock()
_exposure_state = {"enabled": MANUAL_EXPOSURE_ENABLED, "value": MANUAL_EXPOSURE_VALUE}


def apply_exposure(enabled, value):
    """Reach into V4LCamera's private `_cap` (a cv2.VideoCapture) to set V4L2
    exposure controls directly - no public Camera/BaseCamera/V4LCamera API
    exposes exposure/brightness/gain (confirmed by reading the installed
    library source). Never raises - logs a warning and leaves the camera on
    whatever exposure mode it already had if `_cap` doesn't exist or behave
    as expected (e.g. after a future library update changes its internals).
    """
    try:
        camera._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1 if enabled else 3)
        if enabled:
            camera._cap.set(cv2.CAP_PROP_EXPOSURE, value)
        logger.info(f"Exposure: {'manual=' + str(value) if enabled else 'auto'}.")
        return True
    except Exception as e:
        logger.warning(f"Could not set exposure ({e}) - camera._cap may not exist/behave as expected.")
        return False


apply_exposure(_exposure_state["enabled"], _exposure_state["value"])

object_detection = ObjectDetection(confidence=DETECTION_CONFIDENCE)
logger.info("Camera and object-detection model ready.")

_calib = None
if CALIBRATION_PATH.exists():
    _calib = homography.load_calibration(str(CALIBRATION_PATH))
    expected_size = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    logger.info(f"Loaded homography calibration (rmse={_calib['rmse']:.1f}mm).")
    if _calib["image_size"] != expected_size:
        logger.warning(f"Calibration image_size {_calib['image_size']} != runtime {expected_size} - re-run calibrate.py.")
else:
    logger.warning(f"No calibration.npz at {CALIBRATION_PATH} - autonomous tracking needs it, see calibrate.py.")

HOME_LENGTHS = cable_lengths(HOME_POSITION)
logger.info(f"Cable lengths at HOME_POSITION {HOME_POSITION}: "
            f"{[round(length, 1) for length in HOME_LENGTHS]} mm")

# The last position we commanded the camera to - the sole record of "where
# it currently is", since there's no position sensor to read it back from.
# Starts at HOME_POSITION; every move_to() call below updates it.
current_position = list(HOME_POSITION)

_executor = ThreadPoolExecutor(max_workers=len(MOTOR_SERIALS) + 2)  # +kicker, +camera

# Held for the whole duration of a move_to() call - the tracking cycle's
# verify step depends on nothing else moving the platform between the drop
# and the check, and manual UI moves previously raced with the autonomous
# loop with no synchronization at all. This only prevents two move_to() calls
# from executing *simultaneously*; a manual move issued *between* two cycle
# steps can still interleave - accepted residual gap, see the handoff plan.
_motion_lock = threading.Lock()


def move_to(position):
    with _motion_lock:
        target_lengths = cable_lengths(position)

        def move_one(i, target, home):
            theta = REEL_IN_SIGN[i] * (home - target) / MM_PER_DEGREE
            logger.info(f"Motor {i + 1}: target length {target:.1f}mm -> relative position {theta:.0f} deg")
            # The hub itself decides which way to turn to reach this relative
            # position, so no explicit direction is needed here.
            motors[i].motor_run_to_relative_position(theta, speed=MOTOR_SPEED)

        # Dispatch all 4 winches concurrently (bridge runs threaded) so they move
        # at the same time instead of one after another.
        futures = [
            _executor.submit(move_one, i, target, home)
            for i, (target, home) in enumerate(zip(target_lengths, HOME_LENGTHS))
        ]
        for future in futures:
            future.result()
        current_position[:] = position


def nudge(axis, sign):
    index = AXES.index(axis)
    position = list(current_position)
    position[index] += sign * NUDGE_STEP_MM
    move_to(position)


def jog(motor_index, reel_in):
    """Manually turn one winch by a fixed step, in whichever physical
    direction currently shortens (reel_in=True) or lengthens (False) its
    cable, per that motor's REEL_IN_SIGN. This is a real relative turn from
    wherever the motor physically is, so it stays consistent with the hub's
    own relative-position counter used by move_to() - no separate tracking
    needed."""
    sign = REEL_IN_SIGN[motor_index] if reel_in else -REEL_IN_SIGN[motor_index]
    direction = (
        legoeducation.MOTOR_MOVE_DIRECTION_CLOCKWISE
        if sign > 0
        else legoeducation.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
    )
    motors[motor_index].motor_run_for_degrees(JOG_STEP_DEGREES, direction=direction, speed=JOG_SPEED)


def reset_home():
    """Re-anchor wherever the platform physically is RIGHT NOW as
    HOME_POSITION - the same zeroing done once at startup (see
    connect_motor()/HOME_LENGTHS above), but re-runnable live to recover from
    physical drift (manual jogging, cable slip) without restarting the app.
    HOME_LENGTHS itself doesn't need recomputing - it's a fixed function of
    the HOME_POSITION constant, not of the motors' actual state - only each
    motor's own relative-position zero and current_position need resetting."""
    with _motion_lock:
        for motor in motors:
            motor.motor_reset_relative_position()
        current_position[:] = HOME_POSITION
    logger.info(f"Reset home: all 4 winch motors re-zeroed at their current physical position, now HOME_POSITION {HOME_POSITION}.")


def kick():
    degrees = KICKER_ROTATIONS * 360
    logger.info(f"Kicking: {KICKER_ROTATIONS} rotations at {KICKER_SPEED}% speed.")
    kicker_motor.motor_run_for_degrees(degrees, speed=KICKER_SPEED)


# ── Autonomous tracking: sample -> homography -> direct move -> verify ───
# The camera looks straight down at a floor plane parallel to its sensor, but
# the cycle now needs *absolute* world coordinates (to drive to in one shot,
# not just a relative direction to nudge toward each frame) - see
# tracking_cycle.py and calibrate.py.

_autonomous_lock = threading.Lock()
_autonomous_enabled = False

_status_lock = threading.Lock()
_status = {
    "platform_detected": False,
    "ball_detected": False,
    "autonomous": False,
    "calibrated": _calib is not None,
    "cycle_phase": None,
    "samples_collected": 0,
    "sample_count_target": SAMPLE_COUNT,
    "last_target_mm": None,
    "last_verify_error_mm": None,
    "verify_retry_count": 0,
    "last_kick_ago_s": None,
}

_frame_lock = threading.Lock()
_latest_frame_jpeg = None

_last_kick_time = 0.0

_cycle = tracking_cycle.TrackingCycle(
    sample_count=SAMPLE_COUNT,
    travel_height_mm=TRAVEL_HEIGHT_MM,
    drop_height_mm=DROP_HEIGHT_MM,
    verify_tolerance_mm=VERIFY_TOLERANCE_MM,
    max_verify_retries=MAX_VERIFY_RETRIES,
    stale_timeout_s=LOST_TARGET_TIMEOUT_S,
    kicker_offset_mm=KICKER_OFFSET_MM,
    correction_gain=VERIFY_CORRECTION_GAIN,
    settle_time_s=KICK_SETTLE_S,
)


def set_autonomous(enabled):
    global _autonomous_enabled
    with _autonomous_lock:
        _autonomous_enabled = enabled
        if not enabled:
            _cycle.reset()  # abandon any mid-flight cycle rather than resume blind
    logger.info(f"Autonomous tracking {'ON' if enabled else 'OFF'}.")


def is_autonomous():
    with _autonomous_lock:
        return _autonomous_enabled


def best_detection(detections, label):
    matches = [d for d in detections if d.get("class_name") == label]
    return max(matches, key=lambda d: float(d["confidence"]), default=None)


# calibrate.py's homography came out with both world_x and world_y mirrored
# relative to ANCHORS' convention (confirmed live on both axes: driving to a
# ball at a smaller x/y than the platform moved it toward LARGER x/y instead)
# - likely from which physical edge the measurements in points.csv were taken
# relative to. Flipping here, at the single point where homography output
# enters the system, corrects it without touching cable_lengths()/move_to()
# (used by manual jog/nudge/home, which are already correct) or
# points.csv/calibration.npz themselves. If points.csv is ever re-measured
# with world_x/world_y consistently taken from ANCHORS[0] going toward
# ANCHORS[1]/ANCHORS[3], these flips should be removed.
WORLD_X_EXTENT_MM = ANCHORS[1][0]  # 977.9mm - the x-extent ANCHORS defines
WORLD_Y_EXTENT_MM = ANCHORS[3][1]  # 939.8mm - the y-extent ANCHORS defines


def detection_to_world(det):
    """Convert a detection's bottom-center bbox point to world (x, y) mm via
    the loaded homography. None if there's no detection or no calibration."""
    if det is None or _calib is None:
        return None
    uv = imaging.bottom_center(det["bounding_box_xyxy"])
    x, y = homography.image_to_world(uv, _calib["H"])
    x = WORLD_X_EXTENT_MM - x
    y = WORLD_Y_EXTENT_MM - y
    return x, y


def tracking_loop():
    global _last_kick_time, _latest_frame_jpeg

    loop_start = time.time()
    frame = camera.capture()
    if frame is None:
        time.sleep(LOOP_PERIOD)
        return

    frame = imaging.center_crop(frame, MODEL_INPUT_SIZE)
    pil_image = numpy_to_pil(frame)
    result = object_detection.detect(pil_image)
    detections = (result or {}).get("detection", [])

    platform_det = best_detection(detections, PLATFORM_LABEL)
    ball_det = best_detection(detections, BALL_LABEL)

    autonomous = is_autonomous()
    now = time.time()

    # Each iteration advances the cycle by AT MOST ONE blocking step (one
    # move_to() call, or one detection) - never the whole sample->lift->
    # translate->drop->verify sequence in one go. A prior incident already
    # happened where a single loop iteration blocked too long, got SIGKILL'd
    # mid-shutdown, and left the LEGO BLE bridge in a stuck state requiring a
    # service restart to recover.
    if autonomous and _calib is not None:
        ball_world = detection_to_world(ball_det)
        platform_world = detection_to_world(platform_det)
        action = _cycle.step(tuple(current_position[:2]), ball_world, platform_world, now)

        if action.reason:
            logger.info(f"Tracking cycle: {action.reason}")
        if action.kind == "move":
            move_to(action.target)
        elif action.kind == "kick":
            if (now - _last_kick_time) > KICK_COOLDOWN_S:
                logger.info("Cycle verified - kicking.")
                kick()
                _last_kick_time = now
        elif action.kind == "reset":
            logger.warning(f"Tracking cycle reset: {action.reason}")
    elif autonomous and _calib is None:
        logger.warning("Autonomous tracking is on but no calibration is loaded - see calibrate.py.")

    annotated = draw_bounding_boxes(pil_image, result) if result else pil_image
    overlay_lines = [f"phase: {_cycle.phase}" if autonomous else "autonomous: OFF"]
    if _cycle.target is not None:
        overlay_lines.append(f"target: ({_cycle.target[0]:.0f}, {_cycle.target[1]:.0f})mm")
    if _cycle.last_verify_error_mm is not None:
        overlay_lines.append(f"verify err: {_cycle.last_verify_error_mm:.1f}mm")
    ImageDraw.Draw(annotated).text((10, 10), "\n".join(overlay_lines), fill=(255, 255, 0))
    # Encode JPEG directly (not the Brick's get_image_bytes(), which always
    # produces PNG - much bigger and slower to encode/transfer per frame).
    buf = io.BytesIO()
    annotated.convert("RGB").save(buf, format="JPEG", quality=70)
    with _frame_lock:
        _latest_frame_jpeg = buf.getvalue()

    with _status_lock:
        _status.update(
            platform_detected=platform_det is not None,
            ball_detected=ball_det is not None,
            autonomous=autonomous,
            calibrated=_calib is not None,
            cycle_phase=_cycle.phase,
            samples_collected=len(_cycle.samples),
            sample_count_target=_cycle.sample_count,
            last_target_mm=_cycle.target,
            last_verify_error_mm=_cycle.last_verify_error_mm,
            verify_retry_count=_cycle.verify_retry_count,
            last_kick_ago_s=round(now - _last_kick_time, 1) if _last_kick_time else None,
        )

    elapsed = time.time() - loop_start
    if elapsed < LOOP_PERIOD:
        time.sleep(LOOP_PERIOD - elapsed)


def stream_video_feed():
    def generate():
        while True:
            with _frame_lock:
                frame_jpeg = _latest_frame_jpeg
            if frame_jpeg is None:
                time.sleep(0.02)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_jpeg + b"\r\n"
            time.sleep(0.02)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


ui = WebUI()


def _position_payload():
    return {"x": current_position[0], "y": current_position[1], "z": current_position[2]}


def handle_get_state(client, data):
    ui.send_message("state_update", _position_payload(), client)


def handle_move_to(client, data):
    try:
        x, y = float(data["x"]), float(data["y"])
    except (KeyError, TypeError, ValueError):
        logger.error(f"Rejected malformed move_to message: {data!r}")
        ui.send_message("move_error", {"error": "x and y must both be numbers."}, client)
        return
    z = current_position[2]  # unaffected by this form - only the nudge buttons change z
    logger.info(f"Moving to ({x}, {y}, {z}).")
    move_to((x, y, z))
    ui.send_message("move_done", _position_payload(), client)


def handle_go_home(client, data):
    logger.info(f"Returning to HOME_POSITION {HOME_POSITION}.")
    move_to(HOME_POSITION)
    ui.send_message("go_home_done", _position_payload(), client)


def handle_reset_home(client, data):
    logger.info("Resetting home to the platform's current physical position.")
    reset_home()
    ui.send_message("reset_home_done", _position_payload(), client)


def handle_nudge(client, data):
    axis = data.get("axis")
    if axis not in AXES:
        ui.send_message("nudge_error", {"error": "axis must be x, y, or z."}, client)
        return
    try:
        sign = 1 if float(data["sign"]) >= 0 else -1
    except (KeyError, TypeError, ValueError):
        logger.error(f"Rejected malformed nudge message: {data!r}")
        ui.send_message("nudge_error", {"error": "sign is required."}, client)
        return
    logger.info(f"Nudging {axis} by {sign * NUDGE_STEP_MM:+.1f}mm.")
    nudge(axis, sign)
    ui.send_message("nudge_done", _position_payload(), client)


def handle_jog(client, data):
    try:
        motor_index = int(data["motor"])
        reel_in = bool(data["reel_in"])
    except (KeyError, TypeError, ValueError):
        logger.error(f"Rejected malformed jog message: {data!r}")
        ui.send_message("jog_error", {"error": "motor and reel_in are required."}, client)
        return
    if not 0 <= motor_index < len(motors):
        ui.send_message("jog_error", {"error": f"motor must be 0-{len(motors) - 1}."}, client)
        return
    logger.info(f"Jogging motor {motor_index + 1} {'in' if reel_in else 'out'}.")
    jog(motor_index, reel_in)
    ui.send_message("jog_done", {"motor": motor_index, "reel_in": reel_in}, client)


def handle_kick(client, data):
    try:
        kick()
    except Exception as e:
        logger.exception("Kick failed.")
        ui.send_message("kick_error", {"error": str(e)}, client)
        return
    ui.send_message("kick_done", {}, client)


def handle_get_tracking_state(client, data):
    with _status_lock:
        payload = dict(_status)
    ui.send_message("tracking_state", payload, client)


def handle_set_autonomous(client, data):
    enabled = bool(data.get("enabled"))
    set_autonomous(enabled)
    ui.send_message("autonomous_state", {"autonomous": enabled})  # broadcast to all clients


def handle_get_exposure_state(client, data):
    with _exposure_lock:
        payload = dict(_exposure_state)
    ui.send_message("exposure_state", payload, client)


def handle_set_exposure(client, data):
    try:
        enabled = bool(data.get("enabled", _exposure_state["enabled"]))
        value = int(data.get("value", _exposure_state["value"]))
    except (TypeError, ValueError):
        ui.send_message("exposure_error", {"error": "value must be an integer."}, client)
        return
    value = max(EXPOSURE_MIN, min(EXPOSURE_MAX, value))
    ok = apply_exposure(enabled, value)
    with _exposure_lock:
        _exposure_state["enabled"] = enabled
        _exposure_state["value"] = value
    if ok:
        ui.send_message("exposure_state", dict(_exposure_state))  # broadcast to all clients
    else:
        ui.send_message("exposure_error", {"error": "Could not apply exposure setting."}, client)


ui.on_message("get_state", handle_get_state)
ui.on_message("move_to", handle_move_to)
ui.on_message("go_home", handle_go_home)
ui.on_message("reset_home", handle_reset_home)
ui.on_message("nudge", handle_nudge)
ui.on_message("jog", handle_jog)
ui.on_message("kick", handle_kick)
ui.on_message("get_tracking_state", handle_get_tracking_state)
ui.on_message("set_autonomous", handle_set_autonomous)
ui.on_message("get_exposure_state", handle_get_exposure_state)
ui.on_message("set_exposure", handle_set_exposure)

ui.expose_api("GET", "/video_feed", stream_video_feed)

try:
    App.run(user_loop=tracking_loop)
finally:
    # App.run() only auto-stops registered "bricks" - these LEGO motors and
    # the camera peripheral aren't one, so they're never released on
    # shutdown unless we do it here. This `finally` fires whether run()
    # returns normally or raises (e.g. sys.exit()) after a stop/SIGTERM.
    #
    # Speed matters here: the container runtime SIGKILLs the process if
    # shutdown takes too long, and a killed process leaves the winch/kicker
    # hubs' BLE links dangling from the bridge's point of view - the next
    # app start then times out reconnecting to them (BridgeBusyError) until
    # the stale link expires or ble-bridge.service is restarted. So stop the
    # camera and all 5 motors concurrently, not one after another.
    logger.info("Shutting down - stopping camera and all motors.")

    def stop_and_disconnect(label, motor):
        try:
            motor.stop()
        except Exception:
            logger.exception(f"Failed to stop {label} cleanly.")
        try:
            motor.disconnect()
        except Exception:
            logger.exception(f"Failed to disconnect {label} cleanly.")

    def stop_camera():
        try:
            camera.stop()
        except Exception:
            logger.exception("Failed to stop camera cleanly.")

    futures = [_executor.submit(stop_and_disconnect, f"winch motor {i + 1}", motor) for i, motor in enumerate(motors)]
    futures.append(_executor.submit(stop_and_disconnect, "kicker motor", kicker_motor))
    futures.append(_executor.submit(stop_camera))
    for future in futures:
        future.result()
