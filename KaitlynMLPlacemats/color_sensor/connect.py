"""
connect.py
Attempts to connect to the Color Sensor with several retries, since BLE
connections can be flaky on the first try.

Run:
    python connect.py
"""

import time
import legoeducation as le

MAX_ATTEMPTS = 5
RETRY_DELAY_S = 3

colorsensor = le.ColorSensor()

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"Connection attempt {attempt}/{MAX_ATTEMPTS}...")
    colorsensor.connect()
    if colorsensor.connected:
        print("Connected!")
        break
    print(f"  Not connected. Retrying in {RETRY_DELAY_S}s...")
    time.sleep(RETRY_DELAY_S)
else:
    print("\nFailed to connect after multiple attempts. Checklist:")
    print("  - Is the sensor powered on and its light active?")
    print("  - Is Terminal/your IDE allowed Bluetooth access in macOS Privacy & Security settings?")
    print("  - Is anything else (Coding Canvas, another script) already connected to it?")
    print("  - Has its firmware been updated recently via code.legoeducation.com?")
    raise SystemExit(1)

# ... use colorsensor here ...

colorsensor.disconnect()