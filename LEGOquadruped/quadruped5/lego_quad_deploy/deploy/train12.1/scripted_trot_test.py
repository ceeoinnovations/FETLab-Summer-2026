"""Open-loop diagonal trot on the REAL robot - NO policy, NO sensors.

Purpose: verify LEG_MAP and LEG_SIGNS. This exact gait walks forward in
simulation, so on hardware:
  - walks forward       -> mapping + signs are correct; policy is next
  - walks backward      -> flip ALL four LEG_SIGNS
  - legs pair same-side -> HL/HR (or FL/FR) swapped in LEG_MAP
  - one leg fights the rhythm -> that leg's sign is flipped

    python scripted_trot_test.py     (in the lego-env, like run_on_robot.py)

Ctrl+C stops. Conventions: FL/FR/HL/HR are the ROBOT's left/right -
imagine sitting on the robot facing its forward direction.
"""

import math
import time

import legoeducation as le

# ---- copy these from run_on_robot.py once verified ----
FRONT_CARD = dict(card_color=le.LEGO_COLOR_RED, card_serial="1779")
BACK_CARD = dict(card_color=le.LEGO_COLOR_PURPLE, card_serial="6040")
LEG_MAP = {
  "FL": ("front", "left"),
  "FR": ("front", "right"),
  "HL": ("back", "right"),
  "HR": ("back", "left"),
}
LEG_SIGNS = {"FL": -1, "FR": +1, "HL": -1, "HR": +1}
TEST_SPEED_PCT = 30   # gentle
# --------------------------------------------------------


def connect(card, name):
  dm = le.DoubleMotor()
  dm.connect(**card)
  if not dm.connected:
    raise SystemExit(f"could not connect to {name}")
  print(f"{name} connected")
  return dm


def main():
  front = connect(FRONT_CARD, "front")
  back = connect(BACK_CARD, "back")
  units = {"front": front, "back": back}
  ports = {"left": le.MOTOR_LEFT, "right": le.MOTOR_RIGHT}

  def leg(k):
    u, p = LEG_MAP[k]
    return units[u], ports[p]

  # --- phase alignment: diagonal pairs half a revolution apart ---
  # FL & HR -> 0 deg, FR & HL -> 180 deg (relative phases are what matter)
  print("aligning crank phases...")
  targets = {"FL": 180, "HR": 0, "FR": 0, "HL": 180}
  for k, tgt in targets.items():
    unit, motor = leg(k)
    pos = unit.motor[motor].absolutePosition % 360
    delta = (tgt - pos) % 360
    if delta > 1:
      unit.motor_run_for_degrees(
        int(delta), motor=motor, speed=20,
        direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE)
      time.sleep(1.5)
  print("phases set - starting trot (Ctrl+C to stop)")

  # --- constant-speed trot: equal speeds preserve the phase offsets ---
  try:
    for k in ("FL", "FR", "HL", "HR"):
      unit, motor = leg(k)
      direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if LEG_SIGNS[k] > 0
                   else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
      unit.motor_run(direction=direction, motor=motor, speed=TEST_SPEED_PCT)
    while True:
      time.sleep(0.5)
  except KeyboardInterrupt:
    pass
  finally:
    for k in ("FL", "FR", "HL", "HR"):
      unit, motor = leg(k)
      unit.motor_stop(motor=motor)
    front.disconnect()
    back.disconnect()
    print("\nstopped")


if __name__ == "__main__":
  main()
