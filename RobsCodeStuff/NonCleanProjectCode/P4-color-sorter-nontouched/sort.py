"""
Step 3 — Read colors and sort.

Continuously reads the color sensor, classifies the reading with the
trained KNN, and actuates the double motor to route the object to the
correct bin. If confidence is below the threshold, the motor stops.

Press Ctrl+C to quit.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import joblib
import numpy as np
from config import SERIAL, COLOR_ACTIONS, FEATURES, CONFIDENCE_THRESHOLD
from lelib import colorSensor, doubleMotor

MODEL_FILE = Path(__file__).parent / "color_model.pkl"

# ── Motor actions ─────────────────────────────────────────────────────────────
# (left_speed, right_speed) for each named action
ACTION_MAP = {
    "forward": ( 100,  100),
    "left":    ( -70,  100),
    "right":   ( 100,  -70),
    "stop":    (   0,    0),
}

checkpoint = joblib.load(MODEL_FILE)
pipeline   = checkpoint["pipeline"]
classes    = checkpoint["classes"]
features   = checkpoint["features"]
print(f"Model loaded. Classes: {classes}")

cs = colorSensor()
dm = doubleMotor()
print("Connecting to color sensor...")
cs.connect(SERIAL)
print("Connecting to double motor...")
dm.connect(SERIAL)
print("Connected. Place an object in front of the sensor. Ctrl+C to quit.\n")

# Column header
print(f"{'Color':<12} {'Conf':>5}  {'Action':<10}  {'L':>5}  {'R':>5}")
print("─" * 48)

try:
    while True:
        reading  = cs.raw_reading()
        X        = np.array([[reading[f] for f in features]])
        probs    = pipeline.predict_proba(X)[0]          # fraction of neighbors per class
        idx      = int(np.argmax(probs))
        label    = classes[idx]
        conf     = float(probs[idx])

        if conf >= CONFIDENCE_THRESHOLD:
            action = COLOR_ACTIONS.get(label, "stop")
        else:
            action = "stop"
            label  = f"{label}?"

        left, right = ACTION_MAP.get(action, (0, 0))
        dm.movement_move_tank(left, right)

        print(f"{label:<12} {conf:>5.0%}  {action:<10}  {left:>+5d}  {right:>+5d}", flush=True)
        time.sleep(0.1)

except KeyboardInterrupt:
    pass
finally:
    dm.stop()
    print("\nStopped.")
