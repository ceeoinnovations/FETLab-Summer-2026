"""
Unified multi-model road-race driver.

Connects the motors, color sensor, and camera ONCE, then lets you switch
live between 9 different trained driving approaches by pressing a number
key. Press Q at any time to stop and quit.

  1 — road-race-end-to-end               (end-to-end regression, human demos)
  2 — option1_keypoint_regression         (hardcoded control + learned keypoint)
  3 — option2_object_detection            (hardcoded control + learned grid detector)
  4 — option3_discrete_classification     (learned classifier + lookup table)
  5 — attempt1-road-race-expert-data      (end-to-end regression, trained on #2's output)
  6 — attempt2-road-race-expert-data      (same, more training data)
  7 — attempt3-road-race-expert-data      (same, more training data)
  8 — attempt4-road-race-expert-data      (same, most training data)
  9 — offline_rl_human_data               (TD3+BC offline RL, trained on human joystick data)

All hardware identity (motor card, color-sensor card, camera index) lives
in ONE place: config.py. Edit it there and it applies to every mode.

Obstacle avoidance via the color sensor runs globally, for every mode,
since the sensor is always connected in this unified script.
"""

import statistics
import time
from collections import Counter, deque

import cv2
import torch
from PIL import Image
from torchvision import transforms

from config import (
    SERIAL, SERIAL_COLOR_SENSOR, CAMERA,
    MOTOR_SPEED_MIN, MOTOR_SPEED_MAX, IMG_SIZE, apply_deadzone,
    OBSTACLE_REFLECTION_THRESHOLD, AVOID_BACKUP_SPEED, AVOID_BACKUP_TIME,
    AVOID_TURN_DEGREES, AVOID_DRIVE_SPEED, AVOID_DRIVE_TIME,
    MODE1_MOTOR_MULTIPLIER, ATTEMPTS_MOTOR_MULTIPLIER,
    MODE2_STEER_GAIN, MODE2_STEER_MAX, MODE2_FORWARD_MAX_SPEED,
    MODE2_FORWARD_SLOWDOWN_AREA, MODE2_STOP_AREA_FRACTION,
    MODE2_SEARCH_TURN_SPEED, MODE2_SEARCH_REVERSE_DIRECTION_AFTER,
    MODE2_DETECT_MEDIAN_WINDOW, MODE2_DETECT_EMA_ALPHA, MODE2_VISIBLE_THRESHOLD,
    MODE3_STEER_GAIN, MODE3_STEER_MAX, MODE3_FORWARD_MAX_SPEED,
    MODE3_FORWARD_SLOWDOWN_AREA, MODE3_STOP_AREA_FRACTION,
    MODE3_SEARCH_TURN_SPEED, MODE3_SEARCH_REVERSE_DIRECTION_AFTER,
    MODE3_DETECT_MEDIAN_WINDOW, MODE3_DETECT_EMA_ALPHA, MODE3_CONFIDENCE_THRESHOLD,
    CATEGORIES, CATEGORY_MOTOR_COMMANDS, MODE4_CATEGORY_VOTE_WINDOW,
    MODE9_SEARCH_TURN_SPEED, MODE9_SEARCH_REVERSE_DIRECTION_AFTER,
)
from lelib import doubleMotor, colorSensor
from models.end_to_end_model import build_model as build_end_to_end_model
from models.detector_model import DetectorModel
from models.grid_detector_model import GridDetector, decode_prediction
from models.classifier_model import build_classifier_model
from models.offline_rl_actor_model import build_offline_rl_actor
from color_detect import get_target_color

MODE_NAMES = {
    1: "road-race-end-to-end",
    2: "option1_keypoint_regression",
    3: "option2_object_detection",
    4: "option3_discrete_classification",
    5: "attempt1-road-race-expert-data",
    6: "attempt2-road-race-expert-data",
    7: "attempt3-road-race-expert-data",
    8: "attempt4-road-race-expert-data",
    9: "offline_rl_human_data",
}

WEIGHTS = {
    1: "weights/1_road_race_end_to_end.pt",
    2: "weights/2_option1_keypoint_regression.pt",
    3: "weights/3_option2_object_detection.pt",
    4: "weights/4_option3_discrete_classification.pt",
    5: "weights/5_attempt1.pt",
    6: "weights/6_attempt2.pt",
    7: "weights/7_attempt3.pt",
    8: "weights/8_attempt4.pt",
    9: "weights/9_offline_rl.pt",
}

DEVICE = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")

# Standard MobileNetV2 ImageNet transform, shared by every model (all built
# on the same backbone/input size).
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
# cv2 frames are BGR numpy arrays — this variant accepts one directly via
# ToPILImage(), matching how the original drive.py scripts fed frames in.
FRAME_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def denormalize_speed(v):
    mid = (MOTOR_SPEED_MAX + MOTOR_SPEED_MIN) / 2
    span = (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN) / 2
    return v * span + mid


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


print("Loading all 9 models (this only happens once at startup)...")

end_to_end_models = {}
for mode_num, multiplier in ((1, MODE1_MOTOR_MULTIPLIER),
                             (5, ATTEMPTS_MOTOR_MULTIPLIER),
                             (6, ATTEMPTS_MOTOR_MULTIPLIER),
                             (7, ATTEMPTS_MOTOR_MULTIPLIER),
                             (8, ATTEMPTS_MOTOR_MULTIPLIER)):
    m = load_checkpoint(build_end_to_end_model(), WEIGHTS[mode_num])
    end_to_end_models[mode_num] = {"model": m, "multiplier": multiplier}
    print(f"  loaded mode {mode_num}: {MODE_NAMES[mode_num]}")

keypoint_model = load_checkpoint(DetectorModel(), WEIGHTS[2])
print(f"  loaded mode 2: {MODE_NAMES[2]}")

grid_model = load_checkpoint(GridDetector(), WEIGHTS[3])
print(f"  loaded mode 3: {MODE_NAMES[3]}")

classifier_model = load_checkpoint(build_classifier_model(), WEIGHTS[4])
print(f"  loaded mode 4: {MODE_NAMES[4]}")

offline_rl_actor = load_checkpoint(build_offline_rl_actor(), WEIGHTS[9])
print(f"  loaded mode 9: {MODE_NAMES[9]}")

print("All models loaded.\n")


# ══════════════════════════════════════════════════════════════════════
# Per-mode inference + control functions
# ══════════════════════════════════════════════════════════════════════

def run_end_to_end(mode_num, frame_rgb):
    """Modes 1, 5, 6, 7, 8 — straight regression to [left, right]."""
    entry = end_to_end_models[mode_num]
    tensor = FRAME_TRANSFORM(frame_rgb).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = entry["model"](tensor)[0].cpu()
    left_speed = denormalize_speed(float(pred[0]))
    right_speed = denormalize_speed(float(pred[1]))
    mult = entry["multiplier"]
    left_speed = apply_deadzone(mult * left_speed)
    right_speed = apply_deadzone(mult * right_speed)
    return left_speed, right_speed, None


class ProportionalState:
    """Per-frame smoothing + search state for the hardcoded-control modes
    (2 and 3). Reset whenever the active mode changes or an obstacle
    interrupt fires, so stale readings never leak across a scene change."""

    def __init__(self, median_window):
        self.cx_history = deque(maxlen=median_window)
        self.area_history = deque(maxlen=median_window)
        self.ema_cx = 0.0
        self.ema_area = 0.0
        self.last_turn_sign = 1
        self.search_started_at = None
        self.search_direction = 1


def _smooth(raw_value, history, ema_prev, ema_alpha):
    history.append(raw_value)
    median_value = statistics.median(history)
    return ema_alpha * ema_prev + (1 - ema_alpha) * median_value


def _proportional_control(target, state, *, steer_gain, steer_max,
                           forward_max_speed, forward_slowdown_area,
                           stop_area_fraction, search_turn_speed,
                           search_reverse_after, ema_alpha):
    """Shared control law for modes 2 and 3 — identical logic to the
    original option1/option2 drive.py, parameterized by each mode's own
    tuned constants."""
    if target is not None:
        state.search_started_at = None
        state.ema_cx = _smooth(target["cx_norm"], state.cx_history, state.ema_cx, ema_alpha)
        state.ema_area = _smooth(target["area_frac"], state.area_history, state.ema_area, ema_alpha)

        if state.ema_area >= stop_area_fraction:
            scale = 0.0
        elif state.ema_area <= forward_slowdown_area:
            scale = 1.0
        else:
            scale = (stop_area_fraction - state.ema_area) / (stop_area_fraction - forward_slowdown_area)

        turn = steer_gain * state.ema_cx
        turn = max(-steer_max, min(steer_max, turn)) * scale
        forward = forward_max_speed * scale

        left_speed = forward + turn
        right_speed = forward - turn
        state.last_turn_sign = 1 if state.ema_cx > 0 else -1
        bbox = target["bbox"]
    else:
        if state.search_started_at is None:
            state.search_started_at = time.time()
            state.search_direction = state.last_turn_sign
        elif time.time() - state.search_started_at > search_reverse_after:
            state.search_direction *= -1
            state.search_started_at = time.time()

        left_speed = search_turn_speed * state.search_direction
        right_speed = -search_turn_speed * state.search_direction
        state.cx_history.clear()
        state.area_history.clear()
        bbox = None

    return apply_deadzone(left_speed), apply_deadzone(right_speed), bbox


def run_keypoint(frame_rgb, state):
    """Mode 2 — learned keypoint detector (cx_norm, cy_norm, area_frac,
    visible) feeding the hardcoded proportional controller."""
    pil_img = Image.fromarray(frame_rgb)
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cx_norm, cy_norm, area_frac, visible_logit = keypoint_model(tensor)
        cx_norm, cy_norm, area_frac, visible_logit = (
            cx_norm[0], cy_norm[0], area_frac[0], visible_logit[0]
        )
        visible_prob = torch.sigmoid(visible_logit)

    target = None
    if float(visible_prob) >= MODE2_VISIBLE_THRESHOLD:
        h, w = frame_rgb.shape[:2]
        cx_norm, cy_norm, area_frac = float(cx_norm), float(cy_norm), float(area_frac)
        approx_side = int((area_frac * h * w) ** 0.5)
        cx_px = int(w / 2 + cx_norm * (w / 2))
        cy_px = int(h / 2 + cy_norm * (h / 2))
        bbox = (max(0, cx_px - approx_side // 2), max(0, cy_px - approx_side // 2),
                approx_side, approx_side)
        target = {"cx_norm": cx_norm, "cy_norm": cy_norm, "area_frac": area_frac, "bbox": bbox}

    return _proportional_control(
        target, state,
        steer_gain=MODE2_STEER_GAIN, steer_max=MODE2_STEER_MAX,
        forward_max_speed=MODE2_FORWARD_MAX_SPEED,
        forward_slowdown_area=MODE2_FORWARD_SLOWDOWN_AREA,
        stop_area_fraction=MODE2_STOP_AREA_FRACTION,
        search_turn_speed=MODE2_SEARCH_TURN_SPEED,
        search_reverse_after=MODE2_SEARCH_REVERSE_DIRECTION_AFTER,
        ema_alpha=MODE2_DETECT_EMA_ALPHA,
    )


def run_grid(frame_rgb, state):
    """Mode 3 — learned YOLO-style grid detector feeding the same shape
    of hardcoded proportional controller, with its own tuned constants."""
    h, w = frame_rgb.shape[:2]
    pil_img = Image.fromarray(frame_rgb)
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        confidence, x_offset, y_offset, bw, bh = grid_model(tensor)

    result = decode_prediction(confidence[0], x_offset[0], y_offset[0],
                                bw[0], bh[0], MODE3_CONFIDENCE_THRESHOLD)

    target = None
    if result is not None:
        bbox = (
            int((result["cx_frac"] - result["w_frac"] / 2) * w),
            int((result["cy_frac"] - result["h_frac"] / 2) * h),
            int(result["w_frac"] * w),
            int(result["h_frac"] * h),
        )
        target = {"cx_norm": result["cx_norm"], "cy_norm": result["cy_norm"],
                  "area_frac": result["area_frac"], "bbox": bbox}

    return _proportional_control(
        target, state,
        steer_gain=MODE3_STEER_GAIN, steer_max=MODE3_STEER_MAX,
        forward_max_speed=MODE3_FORWARD_MAX_SPEED,
        forward_slowdown_area=MODE3_FORWARD_SLOWDOWN_AREA,
        stop_area_fraction=MODE3_STOP_AREA_FRACTION,
        search_turn_speed=MODE3_SEARCH_TURN_SPEED,
        search_reverse_after=MODE3_SEARCH_REVERSE_DIRECTION_AFTER,
        ema_alpha=MODE3_DETECT_EMA_ALPHA,
    )


def run_classifier(frame_rgb, vote_history):
    """Mode 4 — no continuous control at all. A trained classifier sorts
    the frame into one of CATEGORIES, majority-voted over the last few
    frames, then looked up directly in CATEGORY_MOTOR_COMMANDS."""
    pil_img = Image.fromarray(frame_rgb)
    tensor = TRANSFORM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = classifier_model(tensor)
        idx = int(logits.argmax(dim=1)[0])
    raw_category = CATEGORIES[idx]

    vote_history.append(raw_category)
    category = Counter(vote_history).most_common(1)[0][0]

    left_speed, right_speed = CATEGORY_MOTOR_COMMANDS[category]
    return apply_deadzone(left_speed), apply_deadzone(right_speed), (raw_category, category)


class SearchOnlyState:
    """Minimal search-fallback state for mode 9 — just enough to spin
    toward wherever the target was last seen and reverse periodically,
    same reflex as modes 2/3's search behavior, without any of their
    detection smoothing (mode 9's actor was trained on raw per-frame
    color-detector output, so its live perception matches that exactly)."""

    def __init__(self):
        self.last_turn_sign = 1
        self.search_started_at = None
        self.search_direction = 1


def run_offline_rl(frame_bgr, state):
    """Mode 9 — TD3+BC offline RL actor trained on human joystick data.
    Takes the compact (cx_norm, cy_norm, area_frac, visible) state from
    the classical color detector — the same detector that generated this
    actor's training pseudo-labels — rather than an image directly.

    The training data had the target visible nearly 100% of the time, so
    the actor has essentially no experience with "target not visible."
    Falls back to the same hardcoded search-and-spin reflex as modes 2/3
    in that case, rather than trust the actor outside its training
    distribution."""
    target = get_target_color(frame_bgr)

    if target is not None:
        state.search_started_at = None
        s = torch.tensor([[target["cx_norm"], target["cy_norm"], target["area_frac"], 1.0]],
                          dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            action = offline_rl_actor(s)[0].cpu()
        left_speed = denormalize_speed(float(action[0]))
        right_speed = denormalize_speed(float(action[1]))
        state.last_turn_sign = 1 if target["cx_norm"] > 0 else -1
        bbox = target["bbox"]
    else:
        if state.search_started_at is None:
            state.search_started_at = time.time()
            state.search_direction = state.last_turn_sign
        elif time.time() - state.search_started_at > MODE9_SEARCH_REVERSE_DIRECTION_AFTER:
            state.search_direction *= -1
            state.search_started_at = time.time()

        left_speed = MODE9_SEARCH_TURN_SPEED * state.search_direction
        right_speed = -MODE9_SEARCH_TURN_SPEED * state.search_direction
        bbox = None

    return apply_deadzone(left_speed), apply_deadzone(right_speed), bbox


def avoid_obstacle(dm):
    """Global interrupt: something is right in front of the color sensor.
    Blind, hardcoded escape maneuver, shared by every mode — a reflex,
    not a decision, so it doesn't touch the camera/model at all."""
    print("Obstacle detected near color sensor — avoiding.")
    dm.stop()
    dm.run(AVOID_BACKUP_SPEED)
    time.sleep(AVOID_BACKUP_TIME)
    dm.turn_left(AVOID_TURN_DEGREES)
    dm.run(AVOID_DRIVE_SPEED)
    time.sleep(AVOID_DRIVE_TIME)
    dm.stop()
    dm.turn_right(AVOID_TURN_DEGREES)
    dm.stop()
    print("Obstacle cleared — resuming.")


# ══════════════════════════════════════════════════════════════════════
# Hardware connection — happens once, shared by every mode
# ══════════════════════════════════════════════════════════════════════

dm = doubleMotor()
print("Connecting to motors...")
dm.connect(SERIAL)
print("Connected.\n")

cs = colorSensor()
print("Connecting to color sensor (used as a proximity sensor)...")
cs.connect(SERIAL_COLOR_SENSOR)
print("Connected.\n")

print(f"Connecting to camera: {CAMERA}")
cap = cv2.VideoCapture(CAMERA)
if not cap.isOpened():
    dm.stop()
    raise RuntimeError(f"Could not open camera: {CAMERA}\n"
                       "Check Camo Studio is running and connected.")
print("Camera connected.\n")

print("=" * 70)
print("Press a number key at any time to (re)select a model:")
for n in range(1, 10):
    print(f"  {n} — {MODE_NAMES[n]}")
print("Press Q to stop and quit.")
print("=" * 70)

# ── Mutable per-mode state, reset whenever the active mode changes ──────
mode = 1
state2 = ProportionalState(MODE2_DETECT_MEDIAN_WINDOW)
state3 = ProportionalState(MODE3_DETECT_MEDIAN_WINDOW)
vote4 = deque(maxlen=MODE4_CATEGORY_VOTE_WINDOW)
state9 = SearchOnlyState()


def reset_mode_state():
    global state2, state3, vote4, state9
    state2 = ProportionalState(MODE2_DETECT_MEDIAN_WINDOW)
    state3 = ProportionalState(MODE3_DETECT_MEDIAN_WINDOW)
    vote4 = deque(maxlen=MODE4_CATEGORY_VOTE_WINDOW)
    state9 = SearchOnlyState()


print(f"Starting in mode {mode}: {MODE_NAMES[mode]}\n")

try:
    while cap.isOpened():
        # ── Interrupt: obstacle check takes priority every frame, in every mode ──
        if cs.reflection() > OBSTACLE_REFLECTION_THRESHOLD:
            avoid_obstacle(dm)
            reset_mode_state()
            continue

        ret, frame = cap.read()
        if not ret:
            print("Lost camera feed.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        debug_info = None

        if mode in end_to_end_models:
            left_speed, right_speed, debug_info = run_end_to_end(mode, frame_rgb)
        elif mode == 2:
            left_speed, right_speed, debug_info = run_keypoint(frame_rgb, state2)
        elif mode == 3:
            left_speed, right_speed, debug_info = run_grid(frame_rgb, state3)
        elif mode == 4:
            left_speed, right_speed, debug_info = run_classifier(frame_rgb, vote4)
        elif mode == 9:
            left_speed, right_speed, debug_info = run_offline_rl(frame, state9)
        else:
            left_speed, right_speed = 0, 0

        dm.movement_move_tank(left_speed, right_speed)

        # ── HUD ──
        hud = frame.copy()
        cv2.putText(hud, f"Mode {mode}: {MODE_NAMES[mode]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)
        cv2.putText(hud, f"L: {left_speed:+5.1f}  R: {right_speed:+5.1f}",
                    (10, hud.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

        if mode in (2, 3, 9) and debug_info is not None:
            x, y, bw, bh = debug_info
            cv2.rectangle(hud, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        elif mode in (2, 3, 9):
            cv2.putText(hud, "TARGET NOT FOUND", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        elif mode == 4 and debug_info is not None:
            raw_category, category = debug_info
            cv2.putText(hud, f"{category}  (raw: {raw_category})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 80), 2)

        cv2.putText(hud, "1-9: switch model   Q: quit", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Autonomous Drive - Unified", hud)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key in (ord(str(n)) for n in range(1, 10)):
            new_mode = int(chr(key))
            if new_mode != mode:
                mode = new_mode
                reset_mode_state()
                print(f"Switched to mode {mode}: {MODE_NAMES[mode]}")

        time.sleep(0.033)  # ~30 Hz control loop

finally:
    dm.stop()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")
