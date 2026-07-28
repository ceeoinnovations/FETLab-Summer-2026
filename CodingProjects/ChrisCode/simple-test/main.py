import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from lelib import doubleMotor, controller

SERIAL = 2279  # change to your Bluetooth card serial number

dm   = doubleMotor()
ctrl = controller()

print("Connecting to double motor...")
dm.connect(SERIAL)
print("Connecting to controller (joystick)...")
ctrl.connect(SERIAL)
print("Connected. Use joysticks to drive. Ctrl+C to quit.")

try:
    while True:
        left  = ctrl.left_position()
        right = ctrl.right_position()
        dm.movement_move_tank(left, right)
        time.sleep(0.05)
finally:
    dm.stop()
    print("Stopped.")
