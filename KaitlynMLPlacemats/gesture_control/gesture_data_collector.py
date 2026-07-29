"""
gesture_data_collector.py

Lets students TEACH the computer each hand gesture by showing it to the
webcam and pressing a key to save examples, just like the color-sensor
"show me examples of red / blue / green" activity.

Gestures (matching the "Gesture-Controlled Robot" activity):
    [1] Open hand   -> Drive forward
    [2] Fist        -> Stop
    [3] Thumbs up   -> Turn right
    [4] Point       -> Turn left

Controls while running:
    1 / 2 / 3 / 4   record one example of that gesture
    s               save all collected examples to gesture_data.json
    c               clear all collected examples (start over)
    q               quit (prompts to save if you have unsaved examples)

Tips for good training data:
    - Record 20-30 examples per gesture.
    - Move your hand a little between examples (closer/farther, tilted,
      different spot in frame) so the model isn't fooled by one exact pose.
    - Try to have a few different students/hands contribute examples if
      this will be used by a whole class.
"""

import json
import os
import cv2

from gesture_features import HandDetector, landmarks_to_feature_vector, draw_landmarks

DATA_FILE = "gesture_data.json"

GESTURES = {
    ord("1"): "open_hand",
    ord("2"): "fist",
    ord("3"): "thumbs_up",
    ord("4"): "point",
}

GESTURE_LABELS_DISPLAY = {
    "open_hand": "Open hand -> Drive forward",
    "fist": "Fist -> Stop",
    "thumbs_up": "Thumbs up -> Turn right",
    "point": "Point -> Turn left",
}


def load_existing_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {label: [] for label in GESTURE_LABELS_DISPLAY}


def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)
    print(f"Saved {sum(len(v) for v in data.values())} total examples to {DATA_FILE}")


def draw_hud(frame, data, last_captured_label=None):
    h, w = frame.shape[:2]
    y = 30
    cv2.putText(frame, "Gesture data collector", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y += 30
    for i, (label, desc) in enumerate(GESTURE_LABELS_DISPLAY.items(), start=1):
        count = len(data.get(label, []))
        color = (0, 255, 0) if label == last_captured_label else (255, 255, 255)
        cv2.putText(frame, f"[{i}] {desc}  ({count} examples)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 28
    y += 5
    cv2.putText(frame, "[s] save   [c] clear all   [q] quit", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 2)


def main():
    data = load_existing_data()
    hands = HandDetector(max_num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam. Check the camera index / permissions.")
        return

    last_captured_label = None
    last_captured_timer = 0
    unsaved_changes = False

    print("Show a gesture to the camera, then press the matching number key to save an example.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror for intuitive left/right
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_found = hands.detect(rgb)

        current_features = None
        if hands_found:
            hand_landmarks = hands_found[0]
            draw_landmarks(frame, hand_landmarks)
            current_features = landmarks_to_feature_vector(hand_landmarks)

        if last_captured_timer > 0:
            last_captured_timer -= 1
        else:
            last_captured_label = None

        draw_hud(frame, data, last_captured_label)
        cv2.imshow("Gesture Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            if unsaved_changes:
                print("You have unsaved examples. Press 's' first if you want to keep them.")
            break
        elif key == ord("s"):
            save_data(data)
            unsaved_changes = False
        elif key == ord("c"):
            data = {label: [] for label in GESTURE_LABELS_DISPLAY}
            unsaved_changes = True
            print("Cleared all collected examples (not yet saved).")
        elif key in GESTURES:
            label = GESTURES[key]
            if current_features is None:
                print("No hand detected -- show your hand to the camera and try again.")
            else:
                data[label].append(current_features.tolist())
                unsaved_changes = True
                last_captured_label = label
                last_captured_timer = 15  # frames to flash green
                print(f"Recorded example for '{label}' ({len(data[label])} total)")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()