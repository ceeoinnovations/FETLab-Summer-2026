"""
classify_color.py
Loads a trained color model (from train_color_model.py) built as
accumulated clusters, and classifies new bricks by comparing a new
reading to each cluster's mean and standard deviation.

Run:
    python classify_color.py
"""

import json
import time
import statistics
import legoeducation as le

MODEL_FILE = "color_model.json"

VALID_SAMPLES_NEEDED = 10
MIN_SEPARATION = 40
SETTLE_S = 0.5
SAMPLE_TIMEOUT_S = 6
STD_FLOOR = 5.0


def load_model():
    try:
        with open(MODEL_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No trained model found ({MODEL_FILE}). Run train_color_model.py first.")
        raise SystemExit(1)


def connect_sensor():
    colorsensor = le.ColorSensor()
    colorsensor.connect()
    if not colorsensor.connected:
        print("Error connecting to Color Sensor.")
        raise SystemExit(1)
    return colorsensor


def take_scan(colorsensor, delay_s=0.05):
    """Collect several good readings for the brick being classified and average them."""
    time.sleep(SETTLE_S)
    good_samples = []
    start = time.time()

    while len(good_samples) < VALID_SAMPLES_NEEDED and (time.time() - start) < SAMPLE_TIMEOUT_S:
        r = colorsensor.sensor.rawRed
        g = colorsensor.sensor.rawGreen
        b = colorsensor.sensor.rawBlue
        reflection = colorsensor.sensor.reflection
        separation = max(r, g, b) - min(r, g, b)
        if reflection > 0 and separation >= MIN_SEPARATION:
            good_samples.append((r, g, b))
        time.sleep(delay_s)

    if not good_samples:
        print("  Warning: never got a stable reading -- hold the brick closer/steadier and try again.")
        return (colorsensor.sensor.rawRed, colorsensor.sensor.rawGreen, colorsensor.sensor.rawBlue)

    n = len(good_samples)
    return (
        sum(s[0] for s in good_samples) / n,
        sum(s[1] for s in good_samples) / n,
        sum(s[2] for s in good_samples) / n,
    )


def compute_stats(samples):
    n = len(samples)
    means = tuple(sum(s[i] for s in samples) / n for i in range(3))
    if n > 1:
        stds = tuple(statistics.stdev([s[i] for s in samples]) for i in range(3))
    else:
        stds = (0.0, 0.0, 0.0)
    return means, stds


def z_distance(point, mean, std, floor=STD_FLOOR):
    total = 0.0
    for i in range(3):
        s = max(std[i], floor)
        total += ((point[i] - mean[i]) / s) ** 2
    return total ** 0.5


def classify(reading, model):
    """Rank every known color's cluster by z-distance from the new reading."""
    ranked = []
    for label, samples in model.items():
        mean, std = compute_stats(samples)
        d = z_distance(reading, mean, std)
        ranked.append((label, d, mean, len(samples)))
    ranked.sort(key=lambda item: item[1])
    return ranked


def main():
    model = load_model()
    colorsensor = connect_sensor()

    try:
        while True:
            input("\nPlace a brick under the sensor, then press Enter (or Ctrl+C to quit)...")
            reading = take_scan(colorsensor)
            ranked = classify(reading, model)
            best_label, best_dist, best_mean, best_n = ranked[0]

            print(f"Reading: ({reading[0]:.1f}, {reading[1]:.1f}, {reading[2]:.1f})")
            print(f"Best guess: {best_label}  (z-distance: {best_dist:.2f}, from {best_n} training samples)")
            print("Full ranking:", [(l, round(d, 2)) for l, d, _, _ in ranked])
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        colorsensor.disconnect()


if __name__ == "__main__":
    main()