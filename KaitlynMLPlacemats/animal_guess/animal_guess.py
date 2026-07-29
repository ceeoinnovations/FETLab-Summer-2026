"""
Cat vs. Dog Classifier - by MOVEMENT SIGNATURE, not color.

Idea: attach the Double Motor underneath (or inside) each built cat/dog model.
When you run the "reveal" motion, the model wiggles/jerks in a way that's
characteristic of that animal (e.g. a cat model might pounce - a quick burst
then stop - while a dog model might trot - steady rhythmic motion).

We record the Double Motor's built-in 6-axis IMU (accelerometer + gyroscope)
during that motion, boil the time-series down to a handful of summary
features, and classify new models with nearest-centroid (same idea as the
brick sorter, just swapping color features for motion features).

Usage:
    python animal_guess.py collect animal1
    python animal_guess.py collect animal2
    python animal_guess.py classify
    python animal_guess.py show
"""

import sys
import json
import time
import statistics
import legoeducation as le

MODEL_FILE = "animal_motion_model.json"

# ---- update these to match your Connection Card ----
CARD_COLOR = le.LEGO_COLOR_BLUE
CARD_SERIAL = "0021"

SAMPLE_SECONDS = 3.0
SAMPLE_INTERVAL = 0.05  # 20 Hz


def connect_motor():
    motor = le.DoubleMotor()
    motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not motor.connected:
        print("Error connecting to Double Motor.")
        sys.exit(1)
    return motor


def run_reveal_motion_and_record(motor):
    """Drive a short signature motion while sampling the IMU."""
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []
    yaws = []

    # Kick off a short forward-and-back wiggle. Feel free to change this to
    # whatever motion best reveals the model's character (e.g. a quick pounce
    # vs. a slow steady roll).
    motor.movement_move_for_time(time_ms=400, direction=le.MOVEMENT_DIRECTION_FORWARD, speed=60)

    samples = int(SAMPLE_SECONDS / SAMPLE_INTERVAL)
    for _ in range(samples):
        imu = motor.imu_device
        accel_x.append(imu.accelerometerX)
        accel_y.append(imu.accelerometerY)
        accel_z.append(imu.accelerometerZ)
        gyro_x.append(imu.gyroscopeX)
        gyro_y.append(imu.gyroscopeY)
        gyro_z.append(imu.gyroscopeZ)
        yaws.append(imu.yaw)
        time.sleep(SAMPLE_INTERVAL)

    motor.motor_stop()
    return {
        "accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z,
        "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z,
        "yaw": yaws,
    }


def features_from_raw(raw):
    """Turn a raw time-series capture into a fixed-length feature vector."""
    def stats(series):
        return [
            statistics.mean(series),
            statistics.pstdev(series) if len(series) > 1 else 0.0,
            max(series) - min(series),
        ]

    feats = []
    for key in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "yaw"]:
        feats.extend(stats(raw[key]))
    return feats


def load_model():
    try:
        with open(MODEL_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_model(model):
    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)


def collect(label):
    motor = connect_motor()
    print(f"Get ready - running the reveal motion for the '{label}' model in 2 seconds...")
    time.sleep(2)
    raw = run_reveal_motion_and_record(motor)
    motor.disconnect()

    feats = features_from_raw(raw)
    model = load_model()
    model.setdefault(label, []).append(feats)
    save_model(model)
    print(f"Recorded example #{len(model[label])} for '{label}'.")


def centroid(vectors):
    n = len(vectors[0])
    return [statistics.mean(v[i] for v in vectors) for i in range(n)]


def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def classify():
    model = load_model()
    if len(model) < 2:
        print("Need at least two labels with examples. Run 'collect cat' and 'collect dog' first.")
        return

    centroids = {label: centroid(vectors) for label, vectors in model.items()}

    motor = connect_motor()
    print("Get ready - running the reveal motion on the NEW model in 2 seconds...")
    time.sleep(2)
    raw = run_reveal_motion_and_record(motor)
    motor.disconnect()

    feats = features_from_raw(raw)
    ranked = sorted(centroids.items(), key=lambda item: distance(feats, item[1]))
    print("\nPrediction ranking (closest first):")
    for label, c in ranked:
        print(f"  {label}: distance={distance(feats, c):.2f}")
    print(f"\n--> This model looks most like a: {ranked[0][0].upper()}")


def show():
    model = load_model()
    for label, vectors in model.items():
        print(f"{label}: {len(vectors)} example(s)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "collect" and len(sys.argv) == 3:
        collect(sys.argv[2])
    elif cmd == "classify":
        classify()
    elif cmd == "show":
        show()
    else:
        print(__doc__)