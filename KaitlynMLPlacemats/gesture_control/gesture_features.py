"""
gesture_features.py

Shared helper used by both the data collector and the live controller.

NOTE: MediaPipe removed the old "Solutions" API (`mp.solutions.hands`)
starting with pip version 0.10.31 -- if you installed mediapipe fresh,
you only have the new "Tasks" API available. This file uses the new
API (`mediapipe.tasks`), which downloads a small model file
(hand_landmarker.task, ~10MB) the first time you run it and caches it
locally.

Converts a MediaPipe hand-landmark detection into a small, scale- and
position-invariant feature vector that can be compared with simple
distance math (no deep learning needed). This is the same trick used
for the color-sensor "unsupervised learning" activity: turn a raw
sensor reading into numbers, then compare new numbers to stored
examples.

Feature vector = 21 landmarks x (x, y, z), each landmark:
  1. shifted so the wrist (landmark 0) is the origin
  2. divided by a "hand size" scale (wrist -> middle-finger-MCP distance)

That makes the features roughly the same whether a student's hand is
close to the camera or far away, and roughly the same regardless of
where in the frame the hand is.
"""

import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

NUM_LANDMARKS = 21
FEATURE_LENGTH = NUM_LANDMARKS * 3  # x, y, z per landmark

MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# The 21-point hand skeleton connections (same layout MediaPipe's old
# drawing_utils used -- hardcoded here since the new Tasks API doesn't
# ship a drawing helper).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
)


def _ensure_model_downloaded(model_path):
    if os.path.exists(model_path):
        return
    print(f"Downloading hand landmark model to '{model_path}' (one-time, ~10MB)...")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    print("Download complete.")


class HandDetector:
    """Wraps the MediaPipe Tasks HandLandmarker for a live webcam stream.

    VIDEO running mode requires monotonically increasing timestamps,
    which this class tracks automatically -- just call `.detect(rgb_frame)`
    once per webcam frame.
    """

    def __init__(self, model_path=MODEL_FILENAME, max_num_hands=1,
                 min_detection_confidence=0.6, min_tracking_confidence=0.6):
        _ensure_model_downloaded(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=mp_vision.RunningMode.VIDEO,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._start_time = time.time()

    def detect(self, rgb_frame):
        """
        rgb_frame: HxWx3 uint8 RGB numpy array.
        Returns a list of hands; each hand is a list of 21 landmark
        objects with .x .y .z (normalized 0-1). Empty list if no hand found.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return result.hand_landmarks

    def close(self):
        self._landmarker.close()


def landmarks_to_feature_vector(hand_landmarks):
    """
    hand_landmarks: list of 21 landmark objects with .x .y .z (one hand,
    as returned per-hand from HandDetector.detect()).
    Returns None if the landmarks look degenerate (shouldn't normally happen).
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)  # (21, 3)

    wrist = pts[0].copy()
    middle_mcp = pts[9]

    scale = np.linalg.norm(middle_mcp - wrist)
    if scale < 1e-6:
        return None

    normalized = (pts - wrist) / scale
    return normalized.flatten()  # shape (63,)


def draw_landmarks(frame, hand_landmarks):
    """Draw the hand skeleton onto a BGR frame (for visual feedback).
    frame is modified in place. hand_landmarks: list of 21 landmark
    objects with .x .y .z (normalized 0-1)."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (255, 255, 255), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 200, 0), -1)