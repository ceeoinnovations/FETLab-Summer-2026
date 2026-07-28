# ── Change these before running ───────────────────────────────────────────────

# Bluetooth card serial number
SERIAL = 2279

# Colors you want to sort — must match the labels you use in collect.py.
# Map each color name to a motor action:
#   "left"    → turn left  (route to left bin)
#   "right"   → turn right (route to right bin)
#   "forward" → go forward (route to forward bin)
#   "stop"    → stop motors (reject / unknown)
COLOR_ACTIONS = {
    "red":    "right",
    "blue":   "left",
    "green":  "forward",
    "yellow": "right",
    "white":  "stop",
    "black":  "stop",
}

# Features fed to the KNN — all are normalized 0–1 by lelib.raw_reading().
# Remove any you don't want the model to use.
FEATURES = ["rawRed", "rawGreen", "rawBlue", "reflection", "hue", "saturation", "value"]

# How many nearest neighbors to vote (odd number avoids ties)
K = 5

# Minimum fraction of neighbors that must agree to act (e.g. 0.6 = 3 of 5)
CONFIDENCE_THRESHOLD = 0.6
