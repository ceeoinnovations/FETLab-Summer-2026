"""
Step 1 — Collect labeled sensor readings.

Hold each colored object in front of the sensor and press the
corresponding number key to capture a reading. Aim for 20–30
readings per color, varying the distance and angle slightly each time.

Readings are appended to color_data.csv so you can run this script
multiple times to add more samples.

Controls:
  1–N   capture a reading for color N (shown on screen)
  Q     quit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
import tty
import termios
import select
from config import SERIAL, COLOR_ACTIONS, FEATURES
from lelib import colorSensor

DATA_FILE = Path(__file__).parent / "color_data.csv"
COLORS    = list(COLOR_ACTIONS.keys())
KEY_MAP   = {str(i + 1): COLORS[i] for i in range(len(COLORS))}


def read_key():
    """Return a single keypress if one is waiting, otherwise None. No Enter needed."""
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready = select.select([sys.stdin], [], [], 0.05)[0]
        return sys.stdin.read(1) if ready else None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# Write CSV header if file is new
write_header = not DATA_FILE.exists()
csvfile = open(DATA_FILE, "a", newline="")
writer  = csv.DictWriter(csvfile, fieldnames=["label"] + FEATURES)
if write_header:
    writer.writeheader()

cs = colorSensor()
print("Connecting to color sensor...")
cs.connect(SERIAL)
print("Connected.\n")

counts = {c: 0 for c in COLORS}
print("Hold an object in front of the sensor, then press its number key.")
for i, color in enumerate(COLORS):
    print(f"  {i + 1} = {color}")
print("  Q = quit\n")

try:
    while True:
        reading = cs.raw_reading()

        # Live sensor display — updates every loop without waiting for Enter
        r  = int(reading["rawRed"]     * 65535)
        g  = int(reading["rawGreen"]   * 65535)
        b  = int(reading["rawBlue"]    * 65535)
        rf = int(reading["reflection"] * 255)
        h  = int(reading["hue"]        * 65535)
        counts_str = "  ".join(f"{c}:{counts[c]}" for c in COLORS)
        print(f"\rR={r:5d}  G={g:5d}  B={b:5d}  refl={rf:3d}  hue={h:5d}    [{counts_str}]   ",
              end="", flush=True)

        key = read_key()
        if key is None:
            continue
        elif key.lower() == "q":
            break
        elif key in KEY_MAP:
            label = KEY_MAP[key]
            row   = {"label": label} | {f: reading[f] for f in FEATURES}
            writer.writerow(row)
            csvfile.flush()
            counts[label] += 1
            print(f"\n  → saved '{label}'  (total for this session: {counts[label]})")
        else:
            print(f"\n  Unknown key '{key}' — press 1–{len(COLORS)} or Q")

finally:
    csvfile.close()
    print("\n\nCollection complete. Counts:", counts)
    print(f"Data saved to {DATA_FILE}")
