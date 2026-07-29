"""
gesture_robot_control.py

Real-time gesture control for a LEGO Education CS & AI Kit robot built
with the Double Motor.

    Gesture         Robot action
    -------------   -------------------------
    Open hand       Drive forward (continuous, while held)
    Fist            Stop
    Thumbs up       Turn right (a step, per detection)
    Point           Turn left (a step, per detection)

Run gesture_data_collector.py FIRST to create gesture_data.json with
examples of each gesture -- this program has no idea what a "fist"
looks like until you've shown it some.

--- Connecting to the correct robot in a room full of them ---
Every LEGO Education Connection Card has a color + serial number
printed on it. Set CARD_COLOR / CARD_SERIAL below to match your robot's
card so this program connects to YOUR robot instead of a classmate's.
(See https://github.com/LEGO/LEGOEducation/blob/main/connect.md)

--- Testing without a robot connected ---
Run with `python gesture_robot_control.py --no-robot` to see the
camera + gesture classification overlay without needing BLE hardware
connected. Useful for tuning the classifier before wiring up a robot.
"""

import argparse
import sys
import time
from collections import deque, Counter

import cv2

from gesture_features import HandDetector, landmarks_to_feature_vector, draw_landmarks
from gesture_classifier import GestureClassifier

# ---------------------------------------------------------------------------
# CONFIG -- update these to match your Connection Card and preferences
# ---------------------------------------------------------------------------
CARD_COLOR_NAME = "LEGO_COLOR_BLUE"   # e.g. LEGO_COLOR_AZURE, LEGO_COLOR_RED, ...
CARD_SERIAL = "0021"                    # the serial number printed on your card

DRIVE_SPEED = 35        # 0-100, forward driving speed
TURN_SPEED = 30         # 0-100, turning speed
TURN_STEP_DEGREES = 45  # how far one "turn" gesture turns the robot

CONFIDENCE_THRESHOLD = 0.6   # min agreement among nearest neighbors to trust a prediction
STABLE_FRAMES_REQUIRED = 5   # gesture must be the top vote for this many recent frames
VOTE_WINDOW = 8              # how many recent frames feed the majority vote
TURN_COOLDOWN_SECONDS = 0.8  # minimum time between two turn commands

GESTURE_TO_ACTION_LABEL = {
    "open_hand": "DRIVE FORWARD",
    "fist": "STOP",
    "thumbs_up": "TURN RIGHT",
    "point": "TURN LEFT",
}


class RobotController:
    """Thin wrapper around the LEGO Education Double Motor so the rest of
    this file doesn't need to think about BLE connection details."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.doublemotor = None
        self.le = None
        self.current_action = None  # tracks what the robot is currently doing

        if not self.enabled:
            print("Running with --no-robot: gestures will be classified but no BLE commands sent.")
            return

        import legoeducation as le
        self.le = le

        card_color = getattr(le, CARD_COLOR_NAME)

        self.doublemotor = le.DoubleMotor()
        print(f"Connecting to Double Motor (card color={CARD_COLOR_NAME}, serial={CARD_SERIAL})...")
        self.doublemotor.connect(card_color=card_color, card_serial=CARD_SERIAL)

        if not self.doublemotor.connected:
            print("Error connecting to the Double Motor. Check that it's powered on and broadcasting.")
            sys.exit(1)

        print("Connected!")

    def apply_gesture(self, gesture_label):
        """Called every time we have a confident, stable gesture reading."""
        if gesture_label == "open_hand":
            self._drive_forward()
        elif gesture_label == "fist":
            self._stop()
        elif gesture_label == "thumbs_up":
            self._turn("RIGHT")
        elif gesture_label == "point":
            self._turn("LEFT")

    def _drive_forward(self):
        if self.current_action == "forward":
            return  # already driving, nothing new to send
        self.current_action = "forward"
        if not self.enabled:
            return
        self.doublemotor.movement_move(direction=self.le.MOVEMENT_DIRECTION_FORWARD, speed=DRIVE_SPEED)

    def _stop(self):
        if self.current_action == "stopped":
            return
        self.current_action = "stopped"
        if not self.enabled:
            return
        self.doublemotor.movement_stop()

    def _turn(self, side):
        # Turning is a discrete step, not a held state -- always allowed
        # through (subject to the cooldown applied by the caller).
        self.current_action = "turning"
        if not self.enabled:
            return
        direction = self.le.MOVEMENT_TURN_DIRECTION_RIGHT if side == "RIGHT" else self.le.MOVEMENT_TURN_DIRECTION_LEFT
        self.doublemotor.movement_turn_for_degrees(TURN_STEP_DEGREES, direction=direction)
        # after completing the turn step, forget the "turning" state so a
        # student can chain "point -> point -> open hand" naturally
        self.current_action = None

    def shutdown(self):
        if not self.enabled or self.doublemotor is None:
            return
        try:
            self.doublemotor.movement_stop()
            self.doublemotor.disconnect()
        except Exception as exc:
            print(f"Error during shutdown (safe to ignore if already disconnected): {exc}")


def draw_hud(frame, gesture_label, confidence, stable, robot_enabled):
    h, w = frame.shape[:2]
    action = GESTURE_TO_ACTION_LABEL.get(gesture_label, "...") if gesture_label else "no hand detected"
    color = (0, 255, 0) if stable else (0, 200, 255)

    cv2.putText(frame, f"Gesture: {gesture_label or '-'}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Action: {action}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if gesture_label:
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    mode = "LIVE (robot connected)" if robot_enabled else "PREVIEW (--no-robot)"
    cv2.putText(frame, mode, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)


def main():
    parser = argparse.ArgumentParser(description="Gesture-controlled LEGO Education robot")
    parser.add_argument("--no-robot", action="store_true", help="Preview gesture classification without connecting to BLE hardware")
    parser.add_argument("--data-file", default="gesture_data.json", help="Path to the gesture training data")
    args = parser.parse_args()

    classifier = GestureClassifier(data_file=args.data_file)
    robot = RobotController(enabled=not args.no_robot)
    hands = HandDetector(max_num_hands=1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the webcam.")
        robot.shutdown()
        return

    recent_votes = deque(maxlen=VOTE_WINDOW)
    last_turn_time = 0.0

    print("Show a gesture to drive the robot. Press 'q' in the video window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hands_found = hands.detect(rgb)

            gesture_label = None
            confidence = 0.0
            stable = False

            if hands_found:
                hand_landmarks = hands_found[0]
                draw_landmarks(frame, hand_landmarks)
                features = landmarks_to_feature_vector(hand_landmarks)

                if features is not None:
                    gesture_label, confidence = classifier.predict(features)
                    if confidence >= CONFIDENCE_THRESHOLD:
                        recent_votes.append(gesture_label)
                    else:
                        recent_votes.append(None)
            else:
                recent_votes.append(None)

            # Only act once the same gesture has "won" most of a recent window
            # of frames -- this smooths out flicker from a single bad frame.
            if len(recent_votes) == VOTE_WINDOW:
                vote_counts = Counter(v for v in recent_votes if v is not None)
                if vote_counts:
                    top_label, top_count = vote_counts.most_common(1)[0]
                    if top_count >= STABLE_FRAMES_REQUIRED:
                        stable = True
                        gesture_label = top_label

            if stable:
                if gesture_label in ("thumbs_up", "point"):
                    now = time.time()
                    if now - last_turn_time >= TURN_COOLDOWN_SECONDS:
                        robot.apply_gesture(gesture_label)
                        last_turn_time = now
                        recent_votes.clear()  # avoid re-triggering the same turn repeatedly
                else:
                    robot.apply_gesture(gesture_label)

            draw_hud(frame, gesture_label, confidence, stable, robot.enabled)
            cv2.imshow("Gesture Robot Control", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        robot._stop()
        robot.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()