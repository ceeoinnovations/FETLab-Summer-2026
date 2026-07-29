"""
diagnose_sensor.py (v3)
Streams raw RGB, reflection, and connection status from the Color Sensor.

Adds:
- A "separation" score (max channel - min channel) so you can tell live
  whether a reading is a real color or washed-out ambient light.
- Automatic reconnect attempts if the Bluetooth connection drops.

Run:
    python diagnose_sensor.py
Press Ctrl+C to stop.
"""

import time
import legoeducation as le

SEPARATION_WARNING_THRESHOLD = 40  # below this, channels are too close to trust as a color

colorsensor = le.ColorSensor()
colorsensor.connect()

if not colorsensor.connected:
    print("Error connecting to Color Sensor.")
    raise SystemExit(1)

print("Streaming values. Hold bricks CLOSE (touching, if possible) the sensor.")
print("Press Ctrl+C to stop.\n")

was_connected = True

try:
    while True:
        if not colorsensor.connected:
            if was_connected:
                print(">>> CONNECTION LOST -- attempting to reconnect... <<<")
                was_connected = False
            colorsensor.connect()
            time.sleep(0.5)
            continue
        else:
            if not was_connected:
                print(">>> RECONNECTED <<<")
                was_connected = True

        r = colorsensor.sensor.rawRed
        g = colorsensor.sensor.rawGreen
        b = colorsensor.sensor.rawBlue
        reflection = colorsensor.sensor.reflection
        separation = max(r, g, b) - min(r, g, b)

        flag = "  <-- TOO WASHED OUT, move closer / block ambient light" if separation < SEPARATION_WARNING_THRESHOLD else ""
        print(f"RGB=({r}, {g}, {b})  reflection={reflection}  separation={separation}{flag}")
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\nStopped.")
finally:
    colorsensor.disconnect()