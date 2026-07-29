"""Orient / diagnose the LEGO quadruped motors before running a policy.

    python orient_robot.py    (hardware env; pip install legoeducation numpy)

Why: run_on_robot flails when the policy's 4 coordinated crank commands are
mis-mapped to the physical motors - wrong LEG_MAP (which motor is which leg),
wrong LEG_SIGNS (spin direction), a dead motor, or an encoder that doesn't track
(the policy reads crank angle as feedback). This tool checks all of those, using
the SAME notification-cached absolutePosition the policy uses, so what it reports
is exactly what the policy sees.

Menu:
  1. Sweep each motor  - jog each of the 4 channels CW, report encoder delta
       (responds? which sign?) and you note which physical leg moved and how.
       Builds LEG_MAP + tells you each channel's encoder sign.
  2. Live encoders     - print all 4 absolutePositions continuously; hand-turn
       each crank to see which channel tracks it and in what sign.
  3. Jog one motor     - pick a channel + direction, watch it.
  q. quit (stops all motors).

Then: put the resulting LEG_MAP into run_on_robot12.py, and use scripted_trot_
test.py (open-loop trot) to confirm the HARDWARE can walk at all before blaming
the policy.
"""
import threading
import time

import legoeducation as le

FRONT_CARD = dict(card_color=le.LEGO_COLOR_RED, card_serial="1779")
BACK_CARD = dict(card_color=le.LEGO_COLOR_PURPLE, card_serial="6040")

# the four physical motor channels: (label, unit, port_const, mask)
CHANNELS = [
  ("front-left",  "front", le.MOTOR_LEFT,  1),
  ("front-right", "front", le.MOTOR_RIGHT, 2),
  ("back-left",   "back",  le.MOTOR_LEFT,  1),
  ("back-right",  "back",  le.MOTOR_RIGHT, 2),
]


class UnitState:
  def __init__(self):
    self.lock = threading.Lock()
    self.abspos = {1: 0.0, 2: 0.0}
    self.speed = {1: 0.0, 2: 0.0}

  def callback(self, data):
    for item in le.device_notification_parser(data):
      if type(item).__name__ == "MotorNotification":
        m = item.motorBitMask
        if m in (1, 2):
          with self.lock:
            self.abspos[m] = item.absolutePosition
            self.speed[m] = item.speed


def connect(card, name, state):
  dm = le.DoubleMotor()
  dm.connect(**card)
  if not dm.connected:
    raise SystemExit(f"could not connect to {name} double motor")
  dm.set_notification_callback(state.callback)
  dm.device_notification_request(50)
  print(f"{name} connected")
  return dm


def main():
  fstate, bstate = UnitState(), UnitState()
  front = connect(FRONT_CARD, "front", fstate)
  back = connect(BACK_CARD, "back", bstate)
  units = {"front": front, "back": back}
  states = {"front": fstate, "back": bstate}
  time.sleep(0.6)   # let first notifications arrive

  def getpos(unit, mask):
    st = states[unit]
    with st.lock:
      return st.abspos[mask]

  def sweep():
    print("\n--- sweep each motor (CW 2 s at speed 25) ---")
    print("watch WHICH leg moves and which way the crank/foot goes.\n")
    result = {}
    for label, unit, port, mask in CHANNELS:
      input(f"press Enter to jog {label} ({unit} unit)...")
      p0 = getpos(unit, mask)
      units[unit].motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE,
                            motor=port, speed=25)
      time.sleep(2.0)
      units[unit].motor_stop(motor=port)
      time.sleep(0.4)
      p1 = getpos(unit, mask)
      delta = p1 - p0
      health = "OK" if abs(delta) > 2 else "!! NO ENCODER CHANGE (dead/unmapped?)"
      print(f"   encoder {p0:.0f} -> {p1:.0f}  (delta {delta:+.0f} deg)  [{health}]")
      leg = input("   which physical leg moved? [fl/fr/hl/hr, or blank] ").strip().upper()
      if leg:
        result[leg] = (unit, "left" if port == le.MOTOR_LEFT else "right", delta)
    print("\n--- suggested LEG_MAP (from what you observed) ---")
    print("LEG_MAP = {")
    for k in ("FL", "FR", "HL", "HR"):
      u, p, d = result.get(k, ("???", "???", 0))
      print(f'  "{k}": ("{u}", "{p}"),    # CW encoder delta {d:+.0f}')
    print("}")
    print("note: if any channel said NO ENCODER CHANGE, that motor is the "
          "flail culprit - check the port/cable before anything else.")

  def live():
    print("\n--- live encoders (Ctrl+C to stop) --- hand-turn each crank ---")
    try:
      while True:
        vals = "  ".join(f"{lbl}:{getpos(u, m):7.0f}"
                         for lbl, u, _, m in CHANNELS)
        print("\r" + vals + "   ", end="")
        time.sleep(0.1)
    except KeyboardInterrupt:
      print()

  def jog():
    for i, (lbl, *_) in enumerate(CHANNELS):
      print(f"  [{i}] {lbl}")
    i = int(input("channel #: "))
    d = input("direction [cw/ccw]: ").strip().lower()
    lbl, unit, port, mask = CHANNELS[i]
    direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if d == "cw"
                 else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
    p0 = getpos(unit, mask)
    units[unit].motor_run(direction=direction, motor=port, speed=25)
    time.sleep(2.0)
    units[unit].motor_stop(motor=port)
    time.sleep(0.4)
    print(f"   {lbl}: encoder {p0:.0f} -> {getpos(unit, mask):.0f}")

  try:
    while True:
      choice = input("\n[1] sweep  [2] live encoders  [3] jog one  [q] quit : ").strip()
      if choice == "1":
        sweep()
      elif choice == "2":
        live()
      elif choice == "3":
        jog()
      elif choice == "q":
        break
  finally:
    for label, unit, port, mask in CHANNELS:
      units[unit].motor_stop(motor=port)
    front.disconnect()
    back.disconnect()
    print("stopped")


if __name__ == "__main__":
  main()
