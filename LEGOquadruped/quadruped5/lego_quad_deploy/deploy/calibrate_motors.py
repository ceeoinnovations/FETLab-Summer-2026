"""Calibrate the LEGO double motors before running the policy.

Measures each motor's actual rotation rate (deg/s) at a few speed%
settings, and lets you check each leg's 'forward' spin direction.

    python calibrate_motors.py     (Python 3.14+, pip install legoeducation)

Fill in your two Connection Cards below first.
"""

import time

import legoeducation as le

# ---- your connection cards (from each Double Motor unit) ----
FRONT_CARD = dict(card_color=le.LEGO_COLOR_RED, card_serial="1779")
BACK_CARD = dict(card_color=le.LEGO_COLOR_PURPLE, card_serial="6040")

# The back unit is mounted flipped, so its LEFT/RIGHT are reversed
# relative to the robot. Motors are labeled by ROBOT side below.
BACK_FLIPPED = True


def robot_motors(unit_name):
  """(label, motor_const) pairs in ROBOT left/right terms."""
  flipped = BACK_FLIPPED and unit_name == "back"
  left_const = le.MOTOR_RIGHT if flipped else le.MOTOR_LEFT
  right_const = le.MOTOR_LEFT if flipped else le.MOTOR_RIGHT
  return (("left", left_const), ("right", right_const))


def connect(card, name):
  dm = le.DoubleMotor()
  dm.connect(**card)
  if not dm.connected:
    raise SystemExit(f"could not connect to {name} double motor")
  print(f"{name} connected")
  return dm


def measure(dm, motor, speed_pct, seconds=2.0):
  """Run one motor at speed_pct and measure actual deg/s from position."""
  p0 = dm.motor[motor].position
  dm.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
               motor=motor, speed=speed_pct)
  time.sleep(seconds)
  p1 = dm.motor[motor].position
  dm.motor_stop(motor=motor)
  time.sleep(0.3)
  return (p1 - p0) / seconds


def identify_legs(front, back):
  """Spin each port; you say which physical leg moved. Prints LEG_MAP."""
  print("\n--- identify legs ---")
  print("each port spins for 2 s; type which PHYSICAL leg moved:")
  print("  fl / fr / hl (back-left) / hr (back-right)\n")
  mapping = {}
  for unit_name, dm in (("front", front), ("back", back)):
    for port_name, motor in (("left", le.MOTOR_LEFT), ("right", le.MOTOR_RIGHT)):
      input(f"press Enter to spin {unit_name} unit, {port_name} port...")
      dm.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, motor=motor, speed=30)
      time.sleep(2.0)
      dm.motor_stop(motor=motor)
      leg = input("which leg moved? [fl/fr/hl/hr] ").strip().upper()
      mapping[leg] = (unit_name, port_name)
  print("\npaste this LEG_MAP into run_on_robot.py:\n")
  print("LEG_MAP = {")
  for k in ("FL", "FR", "HL", "HR"):
    u, p = mapping.get(k, ("???", "???"))
    print(f'  "{k}": ("{u}", "{p}"),')
  print("}")
  return mapping


def main():
  front = connect(FRONT_CARD, "front")
  back = connect(BACK_CARD, "back")

  identify_legs(front, back)

  print("\n--- speed calibration (deg/s at each speed%) ---")
  for unit_name, dm in (("front", front), ("back", back)):
    for motor_name, motor in robot_motors(unit_name):
      rates = {pct: measure(dm, motor, pct) for pct in (30, 60, 100)}
      print(f"{unit_name}-{motor_name}: " +
            "  ".join(f"{p}% -> {r:+.0f} deg/s" for p, r in rates.items()))
      print(f"   suggested MAX_DEG_S for this leg: {abs(rates[100]):.0f}")

  print("\n--- direction check ---")
  print("each leg will now spin CLOCKWISE (API sense) for 2 s;")
  print("watch whether the foot sweeps BACKWARD along the ground")
  print("(backward foot sweep = forward robot motion = sign +1)\n")
  for unit_name, dm in (("front", front), ("back", back)):
    for motor_name, motor in robot_motors(unit_name):
      input(f"press Enter to spin {unit_name}-{motor_name}...")
      dm.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, motor=motor, speed=40)
      time.sleep(2.0)
      dm.motor_stop(motor=motor)
      ans = input("did the foot sweep backward (robot would move forward)? [y/n] ")
      print(f"   -> LEG_SIGNS entry for {unit_name}-{motor_name}: "
            f"{+1 if ans.lower().startswith('y') else -1}")

  front.disconnect()
  back.disconnect()
  print("\nput MAX_DEG_S and LEG_SIGNS into run_on_robot.py")


if __name__ == "__main__":
  main()