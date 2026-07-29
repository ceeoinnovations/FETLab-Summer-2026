# pip install mediapipe opencv-python legoeducation

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import ssl
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
from lelib import doubleMotor
from camlib import pick_camera

SERIAL = 1227  # change to your Bluetooth card serial number

# Landmark indices (33-point skeleton, same numbering as the old API)
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_WRIST     = 15
RIGHT_WRIST    = 16

# Skeleton edges to draw on screen
CONNECTIONS = [
    (11, 12),           # shoulders
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), # torso sides
    (23, 24),           # hips
]

# ── Download pose model if needed ─────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "pose_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
if not MODEL_PATH.exists():
    print("Downloading pose model (~3 MB)…")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except urllib.error.URLError:
        # macOS python.org installs often lack root certificates — retry unverified
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(MODEL_URL, context=ctx) as r:
            MODEL_PATH.write_bytes(r.read())
    print("Model ready.")

# ── Build landmarker (VIDEO mode — stable tracking across frames) ─────────────
options = mp_vision.PoseLandmarkerOptions(
    base_options      = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode      = mp_vision.RunningMode.VIDEO,
    num_poses         = 1,
    min_pose_detection_confidence = 0.7,
    min_tracking_confidence       = 0.7,
)
landmarker = mp_vision.PoseLandmarker.create_from_options(options)

# ── Connect to motor ───────────────────────────────────────────────────────────
dm = doubleMotor()
print("Connecting to double motor…")
dm.connect(SERIAL)
print("Connected. Raise/lower wrists to drive. Press Q to quit.")

# ── Open camera ───────────────────────────────────────────────────────────────
cap, start_ms = pick_camera()


def wrist_to_speed(wrist_y, shoulder_y):
    """
    Wrist above shoulder (smaller y in image coords) → positive (forward).
    Dead zone of ±0.08 normalised units to avoid drift at rest.
    """
    offset = shoulder_y - wrist_y
    DEAD = 0.08
    if abs(offset) < DEAD:
        return 0
    speed = (offset - (DEAD if offset > 0 else -DEAD)) * 100
    return max(-100, min(100, int(speed)))


def draw_pose(frame, landmarks, mirrored=False):
    """
    If `mirrored` is True, the frame being drawn on has already been
    horizontally flipped for display, so landmark x-coordinates (which were
    computed from the original, unflipped frame) must be flipped too so the
    skeleton lines up with the mirrored body on screen.
    """
    h, w = frame.shape[:2]
    if mirrored:
        pts = {i: (w - int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    else:
        pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(landmarks)}
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 200, 60), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 4, (0, 255, 80), -1)


try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # NOTE: detection runs on the RAW (unflipped) frame. MediaPipe's
        # LEFT_*/RIGHT_* landmarks are anatomical labels that assume the
        # subject faces the camera in a normal, unflipped image. Flipping
        # the frame before detection would swap those labels and cause
        # your real left arm to be reported (and driven) as the right side.
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms    = int(time.time() * 1000) - start_ms
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect_for_video(mp_image, ts_ms)

        left_speed = right_speed = 0
        lm = None

        if result.pose_landmarks:
            lm          = result.pose_landmarks[0]
            left_speed  = wrist_to_speed(lm[LEFT_WRIST].y,  lm[LEFT_SHOULDER].y)
            right_speed = wrist_to_speed(lm[RIGHT_WRIST].y, lm[RIGHT_SHOULDER].y)

        dm.movement_move_tank(left_speed, right_speed)

        # Flip only a display copy, so the on-screen view feels like a mirror.
        display = cv2.flip(frame, 1)
        if lm is not None:
            draw_pose(display, lm, mirrored=True)

        h, _ = display.shape[:2]
        cv2.putText(display, f"L: {left_speed:+4d}  R: {right_speed:+4d}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 80), 2)
        cv2.putText(display, "Raise wrists above shoulders to drive forward",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("Pose Drive", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.02)  # cap at ~50 Hz

finally:
    dm.stop()
    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")