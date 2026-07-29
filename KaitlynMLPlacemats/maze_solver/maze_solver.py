"""
LEGO Education CS & AI Kit — Tape Maze Solver
===============================================

A robot learns to drive along a tape-line maze using:
  - RGB color from the Color Sensor (rawRed / rawGreen / rawBlue)
  - Reflected light intensity (reflection, 0-100) from the Color Sensor
  - Tabular Q-learning to learn a driving policy through trial and error

The maze has three distinctly-colored tape regions:
  START  — a colored patch/strip where the robot begins
  PATH   — the tape line color the robot must follow through the maze
  END    — a colored patch/strip marking the goal
Anywhere else (the table/floor) is treated as OFF_TRACK.

--------------------------------------------------------------------------
HOW THIS IS ORGANIZED
--------------------------------------------------------------------------
1. RobotInterface   - thin wrapper around the real `legoeducation` package
                       (Double Motor for driving + Color Sensor for input)
2. ColorClassifier  - nearest-centroid classifier over (R, G, B, reflection)
                       that you train by holding the sensor over each zone
3. QLearningAgent   - standard tabular Q-learning (epsilon-greedy, Q-table
                       persisted to JSON so training can resume)
4. MazeEnv          - turns raw sensor reads into (state, reward, done) for
                       the agent, and turns the agent's actions into motor
                       commands
5. train() / drive_with_policy() - the two run modes
6. main()           - interactive CLI: calibrate -> train -> run

--------------------------------------------------------------------------
HARDWARE / API NOTES (double-checked against legoeducation v1.0.6)
--------------------------------------------------------------------------
- ColorSensor().connect() blocks until connected and populates
  colorsensor.sensor with live fields: .rawRed, .rawGreen, .rawBlue,
  .reflection, .color, .hue, .saturation, .value. These update in the
  background once connected (via device_notification_delay), so reading
  colorsensor.sensor.rawRed etc. always gives the latest value - no manual
  polling call needed.
- DoubleMotor().connect() gives you .movement_move_tank(speed_left,
  speed_right) for differential/tank driving and .movement_stop() to stop.
  Speeds are percentages from -100 to 100.
- Both devices support connect(device_name=...) if you need to target a
  specific hub in a classroom with multiple kits running at once (see the
  DEVICE_NAME_* constants below - fill these in per kit).
--------------------------------------------------------------------------
"""

import json
import os
import random
import time
from collections import deque

import legoeducation as le

# ==========================================================================
# CONFIGURATION — tune these for your maze, your robot, and your classroom
# ==========================================================================

# If you're running several kits in the same room, set these so each robot
# only connects to its own hardware (matches your prior multi-kit BLE
# isolation work). This scopes the Bluetooth search to a specific LEGO
# Education Connection Card, so you don't accidentally connect to a
# neighboring team's motor or sensor. Set either to None to skip that filter.
# CARD_SERIAL should be a string (e.g. "0049") if your card's serial has
# leading zeros.
CARD_COLOR = le.LEGO_COLOR_BLUE    # e.g. le.LEGO_COLOR_BLUE
CARD_SERIAL = "0021"   # e.g. "0021"

DEVICE_SEARCH_TIMEOUT = 10  # seconds to scan for each device - increase if your classroom is busy with BLE traffic

# Driving speeds (percent, -100 to 100)
BASE_SPEED = 15
TURN_SPEED_FAST = 20
TURN_SPEED_SLOW = 10
STEP_TIME_S = 0.15          # base duration for turning/backup actions before re-sensing
FORWARD_STEP_TIME_S = 0.18  # shortened back from 0.30 - a longer blind-forward burst was carrying the robot straight across turns before it could re-sense and react; keep it only slightly longer than STEP_TIME_S rather than double
FORWARD_CHECK_INTERVAL_S = 0.04  # how often to re-check the sensor WHILE driving forward, not just before/after - this is what actually lets a turn get noticed mid-motion instead of only after the whole burst finishes
CONSECUTIVE_OFF_TRACK_LIMIT = 24   # raised again now that a deterministic search sweep (not blind luck) is doing recovery - give it enough steps to actually sweep back and forth and find the line
EPISODE_MAX_STEPS = 400

# --- Classification robustness settings ---
# A single noisy sensor read shouldn't be able to end an episode. The zone
# the agent actually reacts to ("confirmed zone") only changes once the
# same, sufficiently-confident classification has shown up CONFIRM_READS
# times in a row.
CONFIRM_READS = 3
# END gets its own, stricter confirmation requirement. A false "reached END"
# is much more costly than a false PATH/OFF_TRACK blip: it silently ends
# the episode, hands out a huge +20 reward, and reinforces whatever
# behavior (even oscillating in place) happened to produce it. Real
# arrival at END should look like a sustained, stable reading as the robot
# drives onto and stays on the marker - not a handful of scattered reads
# picked up while passing through a color gradient between two other zones.
END_CONFIRM_READS = 6
# Also require the reflection value to be relatively stable across those
# reads (a genuine END reading shouldn't be swinging around), which a brief
# pass through a color transition usually will be.
END_REFLECTION_MAX_SPREAD = 8

# Once off track for this many consecutive reads, stop letting Q-learning
# choose the action and switch to a deterministic alternating left/right
# search sweep instead. A single downward-facing sensor can't tell which
# side it drifted off to, so there's no signal in the state for the agent
# to ever learn a reliable recovery policy from - an alternating sweep is
# a well-known robust way to reacquire a line without needing that
# information at all. Q-learning still handles every other decision.
SEARCH_TRIGGER_STREAK = 3
SEARCH_SWEEP_SWITCH_STEPS = 2  # how many steps to try one direction before flipping
# If the distance to the best-matching centroid and the second-best centroid
# are too close together, the read is ambiguous (this is exactly what causes
# "START looks like END" confusion) - in that case we ignore the read rather
# than trust it.
MIN_CONFIDENCE_MARGIN = 300

VERBOSE = True  # print raw sensor reading, action, and reward every single step - turn off once things are working, it's noisy

# If the raw sensor reading doesn't change at all for this many seconds while
# the robot is supposedly acting, the BLE notification stream has likely
# stopped delivering live updates (a known possible BLE quirk) - flag it
# loudly rather than silently training on a frozen/stale reading.
SENSOR_STALE_WARNING_S = 1.0

# Files where learned data is stored (safe to delete to start fresh)
COLOR_MODEL_PATH = "color_model.json"
Q_TABLE_PATH = "q_table.json"

# Q-learning hyperparameters
ALPHA = 0.2          # learning rate
GAMMA = 0.9           # discount factor
EPSILON_START = 0.9
EPSILON_MIN = 0.15           # raised from 0.05 - keeps meaningful random exploration going long-term
EPSILON_DECAY = 0.97          # slowed from 0.90 - takes ~120 episodes to reach the floor instead of ~28
NUM_TRAINING_EPISODES = 40

# Optimistic initialization: unseen state/action pairs start at this value
# rather than 0.0. Since OFF_TRACK/edge states accumulate negative values
# fairly quickly, an untried action starting at 0.0 can look *worse* than a
# well-known "stay safe near START" behavior, which discourages the agent
# from ever trying it. Starting optimistic makes untried actions look
# attractive until the agent has actually sampled them enough to learn
# otherwise - this directly pushes exploration further from the start area.
OPTIMISTIC_INIT_VALUE = 3.0

# Zone labels used throughout
START, PATH, END, OFF_TRACK = "START", "PATH", "END", "OFF_TRACK"
ZONE_LABELS = [START, PATH, END, OFF_TRACK]

# Actions the agent can take
FORWARD = "FORWARD"
VEER_LEFT = "VEER_LEFT"
VEER_RIGHT = "VEER_RIGHT"
SHARP_LEFT = "SHARP_LEFT"
SHARP_RIGHT = "SHARP_RIGHT"
BACK_UP = "BACK_UP"
ACTIONS = [FORWARD, VEER_LEFT, VEER_RIGHT, SHARP_LEFT, SHARP_RIGHT, BACK_UP]

# When exploring randomly (epsilon-greedy), pick actions with these weights
# instead of uniformly. SHARP_LEFT/SHARP_RIGHT pivot mostly in place and
# BACK_UP actively retreats, so picking uniformly among all 6 actions means
# 5 of 6 random choices barely produce forward progress - which looks
# exactly like "exploring in circles near START" no matter what's been
# learned. Weighting toward FORWARD/VEER lets exploration still try every
# action, just not in equal proportion, so net progress along the path is
# far more likely during random exploration.
EXPLORATION_WEIGHTS = {
    FORWARD: 0.40,
    VEER_LEFT: 0.20,
    VEER_RIGHT: 0.20,
    SHARP_LEFT: 0.075,
    SHARP_RIGHT: 0.075,
    BACK_UP: 0.05,
}


# ==========================================================================
# 1. ROBOT INTERFACE — all direct hardware calls live here
# ==========================================================================

class RobotInterface:
    """Wraps the legoeducation Double Motor + Color Sensor so the rest of
    the program never has to think about BLE directly."""

    def __init__(self, card_color=None, card_serial=None):
        self._card_color = card_color
        self._card_serial = card_serial
        self.motor = self._connect_device(le.DoubleMotor(), "Double Motor")
        self.colorsensor = self._connect_device(le.ColorSensor(), "Color Sensor")

        # Belt-and-suspenders: explicitly (re-)request the notification
        # stream at a fast interval. connect() should already do this by
        # default, but if the very first request silently didn't stick,
        # this catches it rather than reading a permanently-frozen value.
        try:
            self.colorsensor.device_notification_request(50)
        except Exception as e:
            print(f"Warning: could not re-request Color Sensor notifications: {e}")

        self._last_reading = None
        self._last_reading_change_time = time.time()
        self._stale_warned = False

    def _connect_device(self, device, label):
        """Explicitly search first (with a real timeout) then connect to
        the found device, rather than relying on bare connect()'s internal
        scan, which uses a much shorter window. Scoped to card_color/
        card_serial if set, so this only finds YOUR kit's hardware even if
        other kits are active nearby. Prints clear guidance if nothing is
        found."""
        print(f"Searching for {label} ({DEVICE_SEARCH_TIMEOUT}s)...")
        found = device.search(
            timeout=DEVICE_SEARCH_TIMEOUT,
            card_color=self._card_color,
            card_serial=self._card_serial,
        )
        if not found:
            raise RuntimeError(
                f"Could not find a {label} over Bluetooth after {DEVICE_SEARCH_TIMEOUT}s.\n"
                f"Check that: the {label} is powered on/charged, it is NOT already connected "
                f"to the LEGO Education app or another program, it's within a few feet of this "
                f"computer, Bluetooth is on with Terminal/Python granted Bluetooth permission, "
                f"and (if set) CARD_COLOR/CARD_SERIAL actually match the card you're using."
            )
        print(f"Found {label}, connecting...")
        device.connect(device=found[0])
        print(f"Connected to {label}.")
        return device

    def connect(self):
        """Devices are already connected in __init__ (search-then-connect),
        kept as a no-op method so the rest of the program's call sites
        (robot.connect()) don't need to change."""
        pass

    def disconnect(self):
        try:
            self.motor.movement_stop()
        except Exception:
            pass
        self.motor.disconnect()
        self.colorsensor.disconnect()

    def read_sensor(self):
        """Returns (r, g, b, reflection) as plain numbers. Also watches for
        the reading being frozen (unchanged) for too long, which usually
        means the BLE notification stream has stopped delivering live
        updates rather than the robot genuinely holding still that long."""
        s = self.colorsensor.sensor
        reading = (s.rawRed, s.rawGreen, s.rawBlue, s.reflection)
        now = time.time()

        if reading != self._last_reading:
            self._last_reading = reading
            self._last_reading_change_time = now
            self._stale_warned = False
        else:
            stale_seconds = now - self._last_reading_change_time
            if stale_seconds > SENSOR_STALE_WARNING_S and not self._stale_warned:
                print(
                    f"\n*** WARNING: Color Sensor reading hasn't changed in "
                    f"{stale_seconds:.1f}s (stuck at rgb={reading[:3]}, "
                    f"reflection={reading[3]}). Notifications normally arrive "
                    f"every ~50-100ms, so this usually means the BLE "
                    f"notification stream has stopped, NOT that the robot is "
                    f"holding still. Try: power-cycling the Color Sensor, "
                    f"checking its battery, and running a standalone test that "
                    f"only connects the Color Sensor (no motors) and prints "
                    f"readings while you move it by hand. ***\n"
                )
                self._stale_warned = True

        return reading

    def do_action(self, action, step_time=None):
        """Drive for a short burst corresponding to the given action, then
        stop (so the next sensor read reflects the new position)."""
        if step_time is None:
            step_time = FORWARD_STEP_TIME_S if action == FORWARD else STEP_TIME_S

        if action == FORWARD:
            self.motor.movement_move_tank(BASE_SPEED, BASE_SPEED, blocking=False)
        elif action == VEER_LEFT:
            self.motor.movement_move_tank(TURN_SPEED_SLOW, BASE_SPEED, blocking=False)
        elif action == VEER_RIGHT:
            self.motor.movement_move_tank(BASE_SPEED, TURN_SPEED_SLOW, blocking=False)
        elif action == SHARP_LEFT:
            self.motor.movement_move_tank(-TURN_SPEED_FAST, TURN_SPEED_FAST, blocking=False)
        elif action == SHARP_RIGHT:
            self.motor.movement_move_tank(TURN_SPEED_FAST, -TURN_SPEED_FAST, blocking=False)
        elif action == BACK_UP:
            self.motor.movement_move_tank(-BASE_SPEED, -BASE_SPEED, blocking=False)
        else:
            raise ValueError(f"Unknown action: {action}")

        time.sleep(step_time)
        self.motor.movement_stop(blocking=False)

    def drive_forward_monitored(self, classifier):
        """Like do_action(FORWARD), but checks the sensor repeatedly WHILE
        driving instead of only once at the end. A single blind burst can
        carry the robot straight across a turn before it ever gets a
        chance to notice the tape has curved away - checking every
        FORWARD_CHECK_INTERVAL_S and stopping the instant the edge shows
        up is what actually catches a turn in time. Returns the last
        (r, g, b, reflection) reading observed, so the caller doesn't need
        a redundant extra sensor read afterward."""
        self.motor.movement_move_tank(BASE_SPEED, BASE_SPEED, blocking=False)
        elapsed = 0.0
        last_reading = self.read_sensor()

        while elapsed < FORWARD_STEP_TIME_S:
            time.sleep(FORWARD_CHECK_INTERVAL_S)
            elapsed += FORWARD_CHECK_INTERVAL_S
            last_reading = self.read_sensor()
            r, g, b, refl = last_reading
            raw_zone, margin = classifier.classify_with_margin(r, g, b, refl)
            if margin >= MIN_CONFIDENCE_MARGIN and raw_zone == OFF_TRACK:
                break  # found the edge - stop driving blind right now

        self.motor.movement_stop(blocking=False)
        return last_reading

    def stop(self):
        self.motor.movement_stop()


# ==========================================================================
# 2. COLOR CLASSIFIER — nearest-centroid over (R, G, B, reflection)
# ==========================================================================

class ColorClassifier:
    """Learns one centroid per zone label from raw sensor samples, then
    classifies new readings by nearest centroid. Reflection is normalized
    separately from RGB since it's on a 0-100 scale rather than 0-1023,
    so it doesn't get drowned out in the distance calculation."""

    def __init__(self):
        self.samples = {label: [] for label in ZONE_LABELS}
        self.centroids = {}

    def collect_samples(self, robot: RobotInterface, label: str, n: int = 25, delay=0.08):
        print(f"\nHold the color sensor steadily over the {label} zone.")
        input("Press Enter when ready...")
        collected = []
        for i in range(n):
            r, g, b, refl = robot.read_sensor()
            collected.append((r, g, b, refl))
            time.sleep(delay)
        self.samples[label].extend(collected)
        print(f"Collected {n} samples for {label}.")

    def train(self):
        """Average the raw samples per label into one centroid each."""
        for label, pts in self.samples.items():
            if not pts:
                continue
            n = len(pts)
            avg = tuple(sum(p[i] for p in pts) / n for i in range(4))
            self.centroids[label] = avg
        missing = [l for l in ZONE_LABELS if l not in self.centroids]
        if missing:
            print(f"Warning: no samples collected for {missing}. "
                  f"Classification will skip these labels.")

    @staticmethod
    def _distance(a, b):
        # a, b = (r, g, b, reflection). Scale reflection (0-100) up to be
        # comparable in magnitude to raw RGB (roughly 0-1023) so it still
        # contributes meaningfully to the distance.
        rgb_dist = sum((a[i] - b[i]) ** 2 for i in range(3))
        refl_dist = ((a[3] - b[3]) * 8) ** 2
        return rgb_dist + refl_dist

    def classify(self, r, g, b, reflection):
        label, _margin = self.classify_with_margin(r, g, b, reflection)
        return label

    def classify_with_margin(self, r, g, b, reflection):
        """Returns (best_label, margin) where margin = distance to the
        SECOND-closest centroid minus distance to the closest one. A large
        margin means a confident, unambiguous match; a small/near-zero
        margin means the reading sits between two zones and shouldn't be
        trusted on its own (this is what causes e.g. START being briefly
        misread as END)."""
        reading = (r, g, b, reflection)
        distances = sorted(
            (self._distance(reading, centroid), label)
            for label, centroid in self.centroids.items()
        )
        if not distances:
            return OFF_TRACK, 0.0
        if len(distances) == 1:
            return distances[0][1], float("inf")
        best_dist, best_label = distances[0]
        second_dist, _ = distances[1]
        margin = second_dist - best_dist
        return best_label, margin

    def report_separation(self):
        """Prints how distinguishable each pair of zones is, and warns if
        any pair is too close together to reliably tell apart. Run this
        right after calibration - if START and END show up as 'TOO CLOSE',
        that's the direct cause of the robot confusing them."""
        labels = list(self.centroids.keys())
        print("\n=== Color separation report ===")
        any_warning = False
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b_label = labels[i], labels[j]
                d = self._distance(self.centroids[a], self.centroids[b_label]) ** 0.5
                flag = ""
                if d < MIN_CONFIDENCE_MARGIN ** 0.5:
                    flag = "  <-- TOO CLOSE, likely to be confused"
                    any_warning = True
                print(f"  {a:10s} vs {b_label:10s}: distance={d:7.1f}{flag}")
        if any_warning:
            print(
                "\nWarning: some zones are hard to tell apart. Try tape colors "
                "that are more visually different (e.g. red/green/blue rather "
                "than two shades of the same color), redo calibration in "
                "consistent lighting, or collect more samples."
            )
        print("================================\n")

    def save(self, path=COLOR_MODEL_PATH):
        data = {"samples": self.samples, "centroids": self.centroids}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved color model to {os.path.abspath(path)}")

    def load(self, path=COLOR_MODEL_PATH):
        if not os.path.exists(path):
            return False
        with open(path) as f:
            data = json.load(f)
        self.samples = {k: [tuple(p) for p in v] for k, v in data["samples"].items()}
        self.centroids = {k: tuple(v) for k, v in data["centroids"].items()}
        print(f"Loaded color model from {os.path.abspath(path)}")
        return True


# ==========================================================================
# 3. Q-LEARNING AGENT
# ==========================================================================

class QLearningAgent:
    """Standard tabular Q-learning with epsilon-greedy exploration.

    State = (current_zone, reflection_bucket, off_track_streak_bucket)
    This keeps the state space small enough to learn from a realistic
    number of real-hardware episodes, while still giving the agent enough
    signal to tell "drifting off the line" from "solidly on the line" and
    to know when it's been off track too long (so it can learn to reverse
    and re-acquire the line rather than drive further away).
    """

    def __init__(self, actions=ACTIONS, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON_START):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = {}  # dict: state_key(str) -> {action: value}
        self.total_episodes_trained = 0  # cumulative across ALL sessions, not just this run

    def _state_key(self, state):
        return json.dumps(state, sort_keys=True)

    def _ensure_state(self, state):
        key = self._state_key(state)
        if key not in self.q:
            self.q[key] = {a: OPTIMISTIC_INIT_VALUE for a in self.actions}
        return key

    def choose_action(self, state, greedy=False):
        key = self._ensure_state(state)
        if not greedy and random.random() < self.epsilon:
            weights = [EXPLORATION_WEIGHTS[a] for a in self.actions]
            return random.choices(self.actions, weights=weights, k=1)[0]
        action_values = self.q[key]
        return max(action_values, key=action_values.get)

    def update(self, state, action, reward, next_state):
        key = self._ensure_state(state)
        next_key = self._ensure_state(next_state)
        best_next = max(self.q[next_key].values())
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[key][action]
        self.q[key][action] += self.alpha * td_error

    def decay_epsilon(self, decay=EPSILON_DECAY, floor=EPSILON_MIN):
        self.epsilon = max(floor, self.epsilon * decay)

    def save(self, path=Q_TABLE_PATH):
        data = {
            "q": self.q,
            "epsilon": self.epsilon,
            "total_episodes_trained": self.total_episodes_trained,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved Q-table ({len(self.q)} states, "
              f"{self.total_episodes_trained} total episodes trained across all sessions) "
              f"to {os.path.abspath(path)}")

    def load(self, path=Q_TABLE_PATH):
        abs_path = os.path.abspath(path)
        if not os.path.exists(path):
            print(f"No existing Q-table found at {abs_path} — starting fresh "
              f"(if you expected prior training to be here, check you're running "
              f"the script from the same directory each time).")
            return False
        with open(path) as f:
            data = json.load(f)
        self.q = data["q"]
        self.epsilon = data.get("epsilon", EPSILON_START)
        self.total_episodes_trained = data.get("total_episodes_trained", 0)
        print(f"Loaded Q-table ({len(self.q)} states, "
              f"{self.total_episodes_trained} total episodes trained so far) "
              f"from {abs_path}")
        return True


# ==========================================================================
# 4. MAZE ENVIRONMENT — bridges sensor readings <-> agent state/reward
# ==========================================================================

def reflection_bucket(reflection):
    """5 buckets instead of 3 - finer resolution here gives the agent a
    chance to learn a graded steering response (e.g. 'reflection creeping
    up toward medium_bright, veer now') instead of only reacting once a
    turn has already fully carried the sensor off the tape."""
    if reflection >= 65:
        return "bright"
    elif reflection >= 45:
        return "medium_bright"
    elif reflection >= 28:
        return "medium"
    elif reflection >= 15:
        return "dark"
    else:
        return "very_dark"


class MazeEnv:
    def __init__(self, robot: RobotInterface, classifier: ColorClassifier):
        self.robot = robot
        self.classifier = classifier
        self.off_track_streak = 0
        self.steps_taken = 0
        self.confirmed_zone = START
        self._pending_zone = None
        self._pending_count = 0
        self._pending_reflections = deque(maxlen=END_CONFIRM_READS)
        self._search_direction = 1  # 1 = veer right first, -1 = veer left first
        self._search_steps_in_direction = 0

    def _off_track_bucket(self):
        if self.off_track_streak == 0:
            return "on_track"
        elif self.off_track_streak < CONSECUTIVE_OFF_TRACK_LIMIT // 2:
            return "briefly_off"
        else:
            return "long_off"

    def _confirm_zone(self, raw_zone, margin, refl):
        """Only lets the confirmed zone change once the same, confident
        classification has repeated enough times in a row. Ambiguous reads
        (low margin) are ignored entirely rather than being allowed to
        reset or advance the streak. END requires extra scrutiny (more
        consecutive reads AND a stable reflection value) since a false
        positive here silently ends the episode with a large reward - a
        brief pass through a color gradient between two other zones can
        otherwise look like a real, if noisy, END reading."""
        if margin < MIN_CONFIDENCE_MARGIN:
            return self.confirmed_zone  # too ambiguous to trust - ignore this read

        if raw_zone == self._pending_zone:
            self._pending_count += 1
        else:
            self._pending_zone = raw_zone
            self._pending_count = 1
            self._pending_reflections.clear()
        self._pending_reflections.append(refl)

        if raw_zone == self.confirmed_zone:
            return self.confirmed_zone

        required_reads = END_CONFIRM_READS if raw_zone == END else CONFIRM_READS
        if self._pending_count < required_reads:
            return self.confirmed_zone

        if raw_zone == END:
            window = list(self._pending_reflections)[-END_CONFIRM_READS:]
            spread = max(window) - min(window)
            if spread > END_REFLECTION_MAX_SPREAD:
                # Reads are bouncing around too much to trust as a genuine,
                # stable arrival at END - likely still just passing through
                # a transition. Keep waiting rather than confirming.
                return self.confirmed_zone

        self.confirmed_zone = raw_zone
        return self.confirmed_zone

    def _read_state(self):
        r, g, b, refl = self.robot.read_sensor()
        raw_zone, margin = self.classifier.classify_with_margin(r, g, b, refl)
        zone = self._confirm_zone(raw_zone, margin, refl)
        state = {
            "zone": zone,
            "reflection": reflection_bucket(refl),
            "off_track": self._off_track_bucket(),
        }
        return state, zone

    def reset(self):
        """Call once the robot has been physically placed back at START."""
        self.off_track_streak = 0
        self.steps_taken = 0
        self.confirmed_zone = START
        self._pending_zone = None
        self._pending_count = 0
        self._pending_reflections.clear()
        self._search_direction = 1
        self._search_steps_in_direction = 0
        state, _ = self._read_state()
        return state

    def forced_recovery_action(self):
        """Returns a deterministic search-sweep action if we've been off
        track long enough to give up on letting Q-learning pick, else None
        (meaning: let the agent choose normally). Alternates sweep
        direction every SEARCH_SWEEP_SWITCH_STEPS steps so it actually
        covers both sides rather than committing to a guess."""
        if self.off_track_streak < SEARCH_TRIGGER_STREAK:
            return None

        self._search_steps_in_direction += 1
        if self._search_steps_in_direction > SEARCH_SWEEP_SWITCH_STEPS:
            self._search_direction *= -1
            self._search_steps_in_direction = 1

        return VEER_RIGHT if self._search_direction > 0 else VEER_LEFT

    def step(self, action):
        if action == FORWARD:
            r, g, b, refl = self.robot.drive_forward_monitored(self.classifier)
        else:
            self.robot.do_action(action)
            r, g, b, refl = self.robot.read_sensor()
        self.steps_taken += 1
        raw_zone, margin = self.classifier.classify_with_margin(r, g, b, refl)
        zone = self._confirm_zone(raw_zone, margin, refl)
        state = {
            "zone": zone,
            "reflection": reflection_bucket(refl),
            "off_track": self._off_track_bucket(),
        }

        done = False
        reward = -0.1  # small per-step penalty encourages efficient paths

        if zone == PATH:
            self.off_track_streak = 0
            self._search_steps_in_direction = 0
            reward += 1.0
        elif zone == START:
            self.off_track_streak = 0
            self._search_steps_in_direction = 0
            reward += 0.2
        elif zone == END:
            self.off_track_streak = 0
            self._search_steps_in_direction = 0
            reward += 20.0
            done = True
        else:  # OFF_TRACK
            self.off_track_streak += 1
            reward -= 1.0  # softened from -2.0 - still discourages drifting, but doesn't punish exploration so harshly that the agent avoids ever leaving the line's immediate vicinity
            if self.off_track_streak >= CONSECUTIVE_OFF_TRACK_LIMIT:
                reward -= 10.0
                done = True  # failed episode - lost the line for too long

        # Closes a reward-hacking loophole: previously, being classified as
        # PATH/START gave the same reward no matter what action produced
        # it - including BACK_UP. That let the agent farm positive reward
        # by rocking back and forth over a colorful boundary without ever
        # making real progress. BACK_UP should only be worthwhile as a
        # genuine recovery move, not a steady-state strategy.
        if action == BACK_UP and not done:
            reward -= 0.5

        if self.steps_taken >= EPISODE_MAX_STEPS:
            done = True

        if VERBOSE:
            print(f"    step {self.steps_taken:3d}  action={action:<12} "
                  f"rgb=({r:.0f},{g:.0f},{b:.0f}) refl={refl:.0f}  "
                  f"raw_zone={raw_zone:<10} margin={margin:6.1f}  "
                  f"confirmed_zone={zone:<10} off_track_streak={self.off_track_streak}  "
                  f"reward={reward:+.2f}")

        return state, reward, done, zone


# ==========================================================================
# 5. TRAIN / RUN LOOPS
# ==========================================================================

def train(env: MazeEnv, agent: QLearningAgent, num_episodes=NUM_TRAINING_EPISODES):
    boost = input(
        f"Current epsilon is {agent.epsilon:.2f} (higher = more random exploration). "
        f"Enter a new value (e.g. 0.6) to boost exploration for this session, "
        f"or press Enter to keep it as-is: "
    ).strip()
    if boost:
        try:
            agent.epsilon = max(0.0, min(1.0, float(boost)))
            print(f"Epsilon set to {agent.epsilon:.2f} for this session.")
        except ValueError:
            print("Not a valid number - keeping current epsilon.")

    for ep in range(1, num_episodes + 1):
        input(f"\n--- Episode {ep}/{num_episodes} --- "
              f"Place the robot at START and press Enter to begin.")
        state = env.reset()
        total_reward = 0.0

        while True:
            action = env.forced_recovery_action() or agent.choose_action(state)
            next_state, reward, done, zone = env.step(action)
            agent.update(state, action, reward, next_state)
            total_reward += reward
            state = next_state

            if done:
                env.robot.stop()
                agent.total_episodes_trained += 1
                outcome = "REACHED END" if zone == END else "lost the line / timed out"
                print(f"Episode {ep} finished: {outcome}. "
                      f"Steps={env.steps_taken}  TotalReward={total_reward:.1f}  "
                      f"Epsilon={agent.epsilon:.2f}  "
                      f"CumulativeEpisodes={agent.total_episodes_trained}")
                break

        agent.decay_epsilon()
        agent.save()  # save after every episode so progress is never lost


def drive_with_policy(env: MazeEnv, agent: QLearningAgent):
    input("\nPlace the robot at START and press Enter to run the learned policy...")
    state = env.reset()
    while True:
        action = env.forced_recovery_action() or agent.choose_action(state, greedy=True)
        state, reward, done, zone = env.step(action)
        if done:
            env.robot.stop()
            if zone == END:
                print("Reached the END zone!")
            else:
                print("Lost the line before reaching the end — "
                      "consider training more episodes.")
            break


# ==========================================================================
# 6. MAIN — interactive CLI
# ==========================================================================

def calibrate(robot: RobotInterface) -> ColorClassifier:
    classifier = ColorClassifier()
    if os.path.exists(COLOR_MODEL_PATH):
        use_existing = input(
            f"Found existing {COLOR_MODEL_PATH}. Use it instead of recalibrating? (y/n): "
        ).strip().lower()
        if use_existing == "y":
            classifier.load()
            classifier.report_separation()
            return classifier

    print("\n=== Color Calibration ===")
    print("You'll hold the sensor over each zone so the robot can learn what it looks like.")
    for label in ZONE_LABELS:
        classifier.collect_samples(robot, label)
    classifier.train()
    classifier.report_separation()
    classifier.save()
    return classifier


def diagnose(robot: RobotInterface, classifier: ColorClassifier):
    """Live-prints raw sensor readings and classification for a few seconds.
    Move the sensor slowly from START to END while this runs to directly see
    whether the two zones are actually distinguishable, and whether the
    classification is stable or flickering."""
    print("\n=== Live diagnostic (10 seconds) ===")
    print("Slowly move the sensor over START, PATH, END, and the floor.")
    print(f"{'R':>5} {'G':>5} {'B':>5} {'Refl':>5}   {'Zone':<10} {'Margin':>8}")
    end_time = time.time() + 10
    while time.time() < end_time:
        r, g, b, refl = robot.read_sensor()
        zone, margin = classifier.classify_with_margin(r, g, b, refl)
        confident = "" if margin >= MIN_CONFIDENCE_MARGIN else "  (ambiguous)"
        print(f"{r:5.0f} {g:5.0f} {b:5.0f} {refl:5.0f}   {zone:<10} {margin:8.1f}{confident}")
        time.sleep(0.2)
    print("=== Diagnostic complete ===\n")


def main():
    print(f"Working directory: {os.getcwd()}")
    print(f"(Q-table and color model files are read/written relative to this "
          f"directory — run the script from the same location every time so "
          f"training accumulates instead of silently starting fresh.)\n")
    robot = RobotInterface(CARD_COLOR, CARD_SERIAL)

    try:
        classifier = calibrate(robot)

        agent = QLearningAgent()
        agent.load()  # resumes prior training if q_table.json exists

        env = MazeEnv(robot, classifier)

        mode = input(
            "\nType 'train' to run training episodes, 'run' to drive using "
            "the current learned policy, or 'diagnose' to watch live sensor "
            "readings and classification: "
        ).strip().lower()

        if mode == "train":
            n = input(f"How many episodes? (default {NUM_TRAINING_EPISODES}): ").strip()
            n = int(n) if n else NUM_TRAINING_EPISODES
            train(env, agent, num_episodes=n)
        elif mode == "diagnose":
            diagnose(robot, classifier)
        else:
            drive_with_policy(env, agent)

    finally:
        robot.disconnect()
        print("Disconnected. Done.")


if __name__ == "__main__":
    main()