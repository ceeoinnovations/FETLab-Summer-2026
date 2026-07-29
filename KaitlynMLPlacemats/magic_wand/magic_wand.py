"""
Magic Wand Gesture Classifier - exploring IMBALANCED datasets.

Attach the Double Motor to your wand. Each person records a "swish" gesture
a number of times of their choosing (e.g. you record 10, a friend records 2).
The program trains a nearest-centroid classifier from whatever's been
recorded, then anyone can test a fresh swish to see who it thinks is
swishing - a great way to *feel* what an imbalanced dataset does to
predictions.

Usage:
    python magic_wand.py record you
    python magic_wand.py record friend
    python magic_wand.py test
    python magic_wand.py show
    python magic_wand.py reset
"""

import sys
import json
import time
import statistics
import legoeducation as le

MODEL_FILE = "wand_gesture_model.json"

CARD_COLOR = le.LEGO_COLOR_BLUE
CARD_SERIAL = "0021"

SAMPLE_SECONDS = 2.0
SAMPLE_INTERVAL = 0.05


def connect_motor():
    motor = le.DoubleMotor()
    motor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not motor.connected:
        print("Error connecting to Double Motor.")
        sys.exit(1)
    return motor


def capture_swish(motor):
    """Sample the IMU while the person performs their wand swish."""
    accel_x, accel_y, accel_z = [], [], []
    gyro_x, gyro_y, gyro_z = [], [], []

    print("Swish now!")
    samples = int(SAMPLE_SECONDS / SAMPLE_INTERVAL)
    for _ in range(samples):
        imu = motor.imu_device
        accel_x.append(imu.accelerometerX)
        accel_y.append(imu.accelerometerY)
        accel_z.append(imu.accelerometerZ)
        gyro_x.append(imu.gyroscopeX)
        gyro_y.append(imu.gyroscopeY)
        gyro_z.append(imu.gyroscopeZ)
        time.sleep(SAMPLE_INTERVAL)

    return {
        "accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z,
        "gyro_x": gyro_x, "gyro_y": gyro_y, "gyro_z": gyro_z,
    }


def features_from_raw(raw):
    def stats(series):
        return [
            statistics.mean(series),
            statistics.pstdev(series) if len(series) > 1 else 0.0,
            max(series) - min(series),
        ]

    feats = []
    for key in ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"]:
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


def centroid(vectors):
    n = len(vectors[0])
    return [statistics.mean(v[i] for v in vectors) for i in range(n)]


def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def record(label):
    motor = connect_motor()
    input(f"Ready to record a swish for '{label}'. Press Enter, then swish...")
    raw = capture_swish(motor)
    motor.disconnect()

    feats = features_from_raw(raw)
    model = load_model()
    model.setdefault(label, []).append(feats)
    save_model(model)
    print(f"Saved example #{len(model[label])} for '{label}'.")


def predict(feats, centroids):
    ranked = sorted(centroids.items(), key=lambda item: distance(feats, item[1]))
    return ranked


def test():
    model = load_model()
    if len(model) < 2:
        print("Need at least two people's gestures recorded first.")
        return
    centroids = {label: centroid(v) for label, v in model.items()}

    motor = connect_motor()
    input("Press Enter, then swish the mystery gesture...")
    raw = capture_swish(motor)
    motor.disconnect()

    feats = features_from_raw(raw)
    ranked = predict(feats, centroids)
    print("\nRanking (closest match first):")
    for label, c in ranked:
        print(f"  {label}: distance={distance(feats, c):.2f}")
    print(f"\n--> My guess: {ranked[0][0]}")


def evaluate():
    """Record several fresh swishes with a KNOWN true label each time, and
    report per-person accuracy - this is the number to compare when you try
    10-vs-10 examples versus 18-vs-2 examples."""
    model = load_model()
    if len(model) < 2:
        print("Need at least two people's gestures recorded first.")
        return
    centroids = {label: centroid(v) for label, v in model.items()}

    results = {label: {"correct": 0, "total": 0} for label in model}
    motor = connect_motor()
    print("Let's run some labeled test swishes. Type 'done' as the label to stop.")
    while True:
        true_label = input("\nWho is about to swish? (label, or 'done'): ").strip()
        if true_label.lower() == "done":
            break
        if true_label not in model:
            print(f"'{true_label}' has no recorded examples yet - skipping.")
            continue
        input("Press Enter, then swish...")
        raw = capture_swish(motor)
        feats = features_from_raw(raw)
        guess = predict(feats, centroids)[0][0]
        correct = guess == true_label
        results[true_label]["total"] += 1
        results[true_label]["correct"] += int(correct)
        print(f"Guessed: {guess}  ({'correct' if correct else 'wrong'})")
    motor.disconnect()

    print("\n--- Per-person accuracy ---")
    for label, r in results.items():
        n_examples = len(model[label])
        acc = (r["correct"] / r["total"] * 100) if r["total"] else float("nan")
        print(f"{label}: trained on {n_examples} example(s) -> "
              f"{r['correct']}/{r['total']} correct ({acc:.0f}%)")


def show():
    model = load_model()
    for label, vectors in model.items():
        print(f"{label}: {len(vectors)} example(s)")


def reset():
    save_model({})
    print("Cleared all recorded gestures.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd == "record" and len(sys.argv) == 3:
        record(sys.argv[2])
    elif cmd == "test":
        test()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "show":
        show()
    elif cmd == "reset":
        reset()
    else:
        print(__doc__)