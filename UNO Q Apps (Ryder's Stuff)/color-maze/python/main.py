import time

from arduino.app_utils import App
import lelib as le
from lelib import doubleMotor, colorSensor

# ── Hardware identifiers ─────────────────────────────────────────
# Replace with the serial numbers printed on your LEGO Bluetooth
# connection cards (see start-here/README.md for how these work).
DRIVE_MOTOR_SERIAL = "7589"
COLOR_SENSOR_SERIAL = "2288"

# ── Tile color -> action mapping ─────────────────────────────────
# Point the sensor at each tile and print(sensor.detect_color()) to
# confirm these match your actual tile colors before relying on them.
GO_COLOR = "Green"
STOP_COLOR = "Red"
LEFT_COLOR = "Orange"
RIGHT_COLOR = "Blue"

DRIVE_SPEED = 30
TURN_DEGREES = 90
FORWARD_STEP_MS = 300   # how long to nudge forward between color checks
CLEAR_MOVE_MS = 500     # extra forward time after a turn, to move off the tile before re-checking it

motor = doubleMotor()
sensor = colorSensor()

print("Color Maze: connecting to hardware...")
motor.connect(card_serial="7589")
sensor.connect(card_serial="2288")
motor.set_speed(DRIVE_SPEED)
print("Color Maze: connected. Starting maze run.")



def handle_tile(color):
    """Reacts to the tile currently under the sensor. Returns True if the maze is solved."""
    print(color)
    if color == STOP_COLOR:
        print("STOP tile detected - halting.")
        motor.stop()
        return True

    if color == LEFT_COLOR:
        print("LEFT tile detected - turning left.")
        motor.stop()
        motor.turn_left(TURN_DEGREES)
        motor.run_time(CLEAR_MOVE_MS)
        motor.stop()
        return False

    if color == RIGHT_COLOR:
        print("RIGHT tile detected - turning right.")
        motor.stop()
        motor.turn_right(TURN_DEGREES)
        motor.run_time(CLEAR_MOVE_MS)
        motor.stop()
        return False

    # GO_COLOR, or floor between tiles (e.g. "No color") - keep driving straight.
    motor.run_time(FORWARD_STEP_MS)
    return False


def loop():
    """Called repeatedly by the App framework."""
    
    color = sensor.detect_color()
    solved = handle_tile(color)


App.run(user_loop=loop)
