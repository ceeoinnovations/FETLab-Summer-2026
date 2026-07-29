"""
train_color_model.py
Builds a color-recognition model as an ACCUMULATED CLUSTER of raw sensor
readings per label, rather than a single overwritten average.

Every time you train a color, all newly collected readings are appended
to whatever samples already exist for that label -- nothing is ever
overwritten or discarded across sessions.

Install first:
    pip install legoeducation

Run:
    python train_color_model.py
"""

import json
import time
import statistics
import legoeducation as le

MODEL_FILE = "color_model.json"

VALID_SAMPLES_NEEDED = 10       # good (non-washed-out) readings to collect per scan
MIN_SEPARATION = 40             # min channel spread to count as a real (non-washed-out) reading
SETTLE_S = 0.5                  # pause after Enter before sampling starts
SAMPLE_TIMEOUT_S = 6            # give up waiting for good readings after this long
SCANS_PER_COLOR = 3             # separate scans per training session


def connect_sensor():
    colorsensor = le.ColorSensor()
    colorsensor.connect()
    if not colorsensor.connected:
        print("Error connecting to Color Sensor.")
        raise SystemExit(1)
    return colorsensor


def collect_valid_samples(colorsensor, count=VALID_SAMPLES_NEEDED, delay_s=0.05):
    """Collect `count` good (non-washed-out) raw (r, g, b) samples."""
    time.sleep(SETTLE_S)
    samples = []
    start = time.time()
    while len(samples) < count and (time.time() - start) < SAMPLE_TIMEOUT_S:
        r = colorsensor.sensor.rawRed
        g = colorsensor.sensor.rawGreen
        b = colorsensor.sensor.rawBlue
        reflection = colorsensor.sensor.reflection
        separation = max(r, g, b) - min(r, g, b)
        if reflection > 0 and separation >= MIN_SEPARATION:
            samples.append((r, g, b))
        time.sleep(delay_s)
    return samples


def compute_stats(samples):
    """samples: list of (r, g, b) -> (mean (r,g,b), std (r,g,b))"""
    n = len(samples)
    means = tuple(sum(s[i] for s in samples) / n for i in range(3))
    if n > 1:
        stds = tuple(statistics.stdev([s[i] for s in samples]) for i in range(3))
    else:
        stds = (0.0, 0.0, 0.0)
    return means, stds


def load_existing_model():
    try:
        with open(MODEL_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def train_color(colorsensor, label, existing_samples):
    """Collect new readings for a color and append them to its existing cluster."""
    new_samples = []
    for i in range(SCANS_PER_COLOR):
        input(f"  Scan {i + 1}/{SCANS_PER_COLOR}: hold the {label} brick steady, then press Enter...")
        batch = collect_valid_samples(colorsensor)
        if not batch:
            print("    Warning: no stable readings captured in this scan -- skipping it.")
            continue
        new_samples.extend(batch)

    if not new_samples:
        print(f"  No usable samples collected for '{label}'; existing data unchanged.")
        return existing_samples

    combined = list(existing_samples) + new_samples
    print(f"  Added {len(new_samples)} new samples. '{label}' cluster now has {len(combined)} total.\n")
    return combined


def main():
    labels_input = input(
        "Enter the color labels to train, separated by commas [red,yellow,blue]: "
    ).strip()
    labels = [l.strip() for l in labels_input.split(",") if l.strip()] or ["red", "yellow", "blue"]

    model = load_existing_model()
    if model:
        print(f"Existing model found with colors: {list(model.keys())}\n")

    colorsensor = connect_sensor()

    try:
        for label in labels:
            print(f"Training '{label}'...")
            existing = model.get(label, [])
            model[label] = train_color(colorsensor, label, existing)
    finally:
        colorsensor.disconnect()

    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)

    print(f"Model saved to {MODEL_FILE}. Summary:")
    for label, samples in model.items():
        mean, std = compute_stats(samples)
        print(
            f"  {label}: n={len(samples)}  "
            f"mean=({mean[0]:.1f}, {mean[1]:.1f}, {mean[2]:.1f})  "
            f"std=({std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f})"
        )


if __name__ == "__main__":
    main()