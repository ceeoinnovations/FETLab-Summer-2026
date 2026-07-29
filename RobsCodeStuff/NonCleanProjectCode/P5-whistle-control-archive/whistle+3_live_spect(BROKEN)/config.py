# ── Change these before running ───────────────────────────────────────────────

# Bluetooth card serial number
SERIAL = 1227

# ── Audio settings ────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100   # Hz — standard mic sample rate
CHUNK_SIZE  = 2048    # samples per FFT window (~46 ms per analysis frame)

# Only frequencies in this range are considered a whistle
WHISTLE_MIN_HZ = 300
WHISTLE_MAX_HZ = 5000

# RMS amplitude below this is treated as silence (0.0–1.0).
# Raise it if background noise triggers false commands.
AMPLITUDE_THRESHOLD = 0.02

# ── Pitch bands → commands ────────────────────────────────────────────────────
# Run calibrate.py to auto-generate pitch_bands.json tuned to your whistle.
# These defaults are used if pitch_bands.json does not exist.
#
# Format: list of (min_hz, max_hz, command)
# Commands: "forward", "backward", "left", "right", "stop"
DEFAULT_PITCH_BANDS = [
    ( 300,  900, "backward"),
    ( 900, 1500, "left"),
    (1500, 2500, "forward"),
    (2500, 5000, "right"),
]

# ── Motor commands: (left_speed, right_speed) ─────────────────────────────────
MOTOR_MAP = {
    "forward":  ( 100,  100),
    "backward": (-100, -100),
    "left":     ( -70,  100),
    "right":    ( 100,  -70),
    "stop":     (   0,    0),
}
