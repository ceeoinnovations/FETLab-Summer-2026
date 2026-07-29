"""Run the trained train12 policy on the REAL LEGO quadruped.

    python run_on_robot12.py   (hardware env; pip install legoeducation numpy)

Copy these three files together onto the hardware machine:
    run_on_robot12.py, numpy_policy.py, policy_weights12.npz

train12 is the FIRST policy trained directly on the faithful MESH model that
actually walks forward (~0.04 m/s, ~1 m/30 s in sim, cold-starts from rest,
clean diagonal trot, never falls). Unlike the older train5.x policies (trained
on the abstract crank model, front-heavy), this one drives ALL FOUR legs in a
real trot, so expect all four motors to cycle.

TWO THINGS DIFFER from the train5.x run_on_robot.py, both essential:

1. PHASE CLOCK IN THE OBSERVATION (obs is 28-dim, was 26). train12 was told the
   trot rhythm via a clock: sin/cos of a phase that advances one cycle every
   CLOCK_T=0.72 s. We reproduce that clock here and append it to the obs. It
   advances on wall-clock time from start, exactly as in training.

2. DIRECTION. train12 walks in the sim's +v_b[0] direction. On this robot that
   is the OPPOSITE of the train5.x "forward" (we flipped the crank convention
   during training). So with the same LEG_SIGNS as train5.3 it will likely walk
   the other way down the body. Options: (a) treat the far end as "front" (you
   said you'd re-face the robot), or (b) deploy train13's policy instead (that
   one was trained for the original direction), or (c) flip ALL four LEG_SIGNS
   here and re-verify. VERIFY DIRECTION SLOWLY FIRST (low MAX_SPEED_PCT).

BEFORE trusting it, re-verify LEG_MAP / LEG_SIGNS with scripted_trot_test.py on
THIS robot (walks backward -> flip all signs; legs pair same-side -> LEG_MAP
swap; one leg fights -> flip that leg's sign). Mesh->hardware crank mapping is
the one thing not verifiable in sim.

IMU + motor data via notification callbacks (real API). Units from device dumps:
  pitch/roll = decidegrees (x0.1 -> deg); gyro = decidegrees/s (x0.1 -> deg/s);
  accel = milli-g -> normalized gravity unit vector; MotorNotification
  motorBitMask: 1 = LEFT, 2 = RIGHT. Ctrl+C stops the motors and exits.
"""

import math
import threading
import time

import numpy as np

import legoeducation as le
from numpy_policy import NumpyPolicy

# ---------------- fill these in ----------------
FRONT_CARD = dict(card_color=le.LEGO_COLOR_RED, card_serial="1779")
BACK_CARD = dict(card_color=le.LEGO_COLOR_PURPLE, card_serial="6040")

COMMAND = 0.03          # commanded forward speed (m/s). train12 range 0.01-0.04.
CLOCK_T = 0.72          # gait-clock period (s) - MUST match training
MAX_DEG_S = 700.0
MAX_SPEED_PCT = 40      # start LOW (~15) the first time, then raise once verified
# per-crank sign, policy/obs order (FL, FR, BL, BR) = mesh GEAR_JOINTS order.
# The policy's DOMINANT crank directions are [+,-,+,-] (measured: FL+ FR- BL+
# BR-), a diagonal trot. LEG_SIGNS maps the policy action -> motor direction
# (CW if action*sign >= 0). With +1s the trot passes straight through as
# [+,-,+,-]. NOTE: the old [-1,+1,-1,+1] * [+,-,+,-] = [-,-,-,-] (all cranks
# one way = the flail) - that was the bug. This walks train12's direction
# (opposite your scripted_trot original). Flip ALL FOUR to walk the other way;
# if ONE leg fights, its LEG_MAP entry (back-unit L/R) is likely swapped.
LEG_SIGNS = np.array([+1.0, +1.0, +1.0, +1.0])
IMU_UNIT = "front"

# Physical unit+port per crank, in policy/obs order (FL, FR, BL, BR). The mesh's
# backLeftGear maps to the physical BACK-RIGHT motor and backRightGear to
# BACK-LEFT (the swapped back-unit labels) - VERIFY on hardware.
LEG_MAP = {
  "FL": ("front", "left"),
  "FR": ("front", "right"),
  "BL": ("back", "right"),
  "BR": ("back", "left"),
}
LEG_ORDER = ("FL", "FR", "BL", "BR")   # matches train12's GEAR_JOINTS order

CONTROL_HZ = 5.0
MAX_CRANK_SPEED = 12.0
TILT_STOP_DEG = 75.0
TILT_STOP_HOLD = 5
USE_LIVE_IMU = True    # train12 trained on live IMU - keep True
GYRO_SIGNS = (1.0, 1.0, 1.0)

# --- heading hold: differential leg speed to hold yaw (train12 curves some) ---
HEADING_HOLD = True
HEADING_UNIT = "front"
HEADING_GAIN = 0.02
HEADING_SIGN = +1.0    # flip to -1 if it steers the wrong way
HEADING_TRIM_MAX = 0.3
DEBUG = True
# ------------------------------------------------

MASK = {"left": 1, "right": 2}
policy = NumpyPolicy("policy_weights12.npz")


class UnitState:
  """Caches the latest notification values for one Double Motor (thread-safe)."""
  def __init__(self):
    self.lock = threading.Lock()
    self.abspos = {1: 0.0, 2: 0.0}
    self.speed = {1: 0.0, 2: 0.0}
    self.pitch = 0.0
    self.roll = 0.0
    self.yaw = 0.0
    self.gyro = (0.0, 0.0, 0.0)
    self.accel = (0.0, 0.0, -1000.0)

  def callback(self, data):
    for item in le.device_notification_parser(data):
      name = type(item).__name__
      with self.lock:
        if name == "MotorNotification":
          m = item.motorBitMask
          if m in (1, 2):
            self.abspos[m] = item.absolutePosition
            self.speed[m] = item.speed
        elif name == "ImuDeviceNotification":
          self.pitch = item.pitch
          self.roll = item.roll
          self.yaw = item.yaw
          self.gyro = (item.gyroscopeX, item.gyroscopeY, item.gyroscopeZ)
          self.accel = (item.accelerometerX, item.accelerometerY,
                        item.accelerometerZ)


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
  state = {"front": fstate, "back": bstate}
  front.imu_set_yaw_face(yaw_face=le.DEVICE_FACE_BACK)
  imu_state = state[IMU_UNIT]

  units = {"front": front, "back": back}
  legs = [(units[LEG_MAP[k][0]], state[LEG_MAP[k][0]], MASK[LEG_MAP[k][1]])
          for k in LEG_ORDER]

  time.sleep(0.5)

  input("\nrotate all four cranks so each FOOT is at its LOWEST point,\n"
        "then press Enter to anchor phase zero... ")
  ang0 = np.array([math.radians(st.abspos[m]) for _, st, m in legs])
  print("phase anchored")

  time.sleep(0.3)
  with imu_state.lock:
    a0 = np.array(imu_state.accel, float)
    pitch0 = imu_state.pitch * 0.1
    roll0 = imu_state.roll * 0.1
  g_ref = a0 / (np.linalg.norm(a0) + 1e-9)

  target = np.array([0.0, 0.0, -1.0])
  v = np.cross(g_ref, target)
  c = float(g_ref @ target)
  if np.linalg.norm(v) < 1e-8:
    R0 = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
  else:
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R0 = np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
  print(f"reference gravity {np.round(g_ref, 2)} -> aligned "
        f"{np.round(R0 @ g_ref, 2)} (want [0 0 -1]); "
        f"pitch0 {pitch0:+.0f} roll0 {roll0:+.0f} deg")

  heading_state = state[HEADING_UNIT]
  with heading_state.lock:
    yaw0 = heading_state.yaw * 0.1
  print(f"reference heading yaw0 = {yaw0:+.0f} deg (point the robot the way "
        f"you want it to go)")

  def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0

  prev_action = np.zeros(4)
  tilt_count = 0
  dt = 1.0 / CONTROL_HZ
  phase_t = 0.0          # gait clock; advances dt each control step (as trained)
  print(f"running (USE_LIVE_IMU={USE_LIVE_IMU}, clock T={CLOCK_T}s) - Ctrl+C to stop")
  try:
    while True:
      t0 = time.time()

      angs, spds = [], []
      for _, st, m in legs:
        with st.lock:
          angs.append(st.abspos[m]); spds.append(st.speed[m])
      with imu_state.lock:
        gyro_raw = np.array(imu_state.gyro, float) * 0.1
        accel = np.array(imu_state.accel, float)

      g_now = accel / (np.linalg.norm(accel) + 1e-9)
      tilt_from_stand = math.degrees(
        math.acos(max(-1.0, min(1.0, float(g_now @ g_ref)))))
      if tilt_from_stand > TILT_STOP_DEG:
        tilt_count += 1
        if tilt_count >= TILT_STOP_HOLD:
          print(f"\ntilt limit ({tilt_from_stand:.0f} deg, sustained) - stopping")
          break
      else:
        tilt_count = 0

      if USE_LIVE_IMU:
        g_b = R0 @ (accel / (np.linalg.norm(accel) + 1e-9))
        gyro = R0 @ (np.radians(gyro_raw) * np.array(GYRO_SIGNS))
      else:
        g_b = np.array([0.0, 0.0, -1.0])
        gyro = np.zeros(3)

      ang = (np.radians(angs) - ang0) * LEG_SIGNS
      vel = np.radians(np.array(spds) / 100.0 * MAX_DEG_S) * LEG_SIGNS

      # phase clock (train12 concept 1): sin/cos of the gait phase
      frac = (phase_t / CLOCK_T) % 1.0
      p = 2.0 * math.pi * frac

      obs = np.concatenate([
        np.zeros(3),          # base lin vel: trained blind
        gyro, g_b,
        np.sin(ang), np.cos(ang),
        vel / 10.0,
        prev_action, [COMMAND],
        [math.sin(p), math.cos(p)],   # 28-dim: the phase clock
      ])

      if HEADING_HOLD:
        with heading_state.lock:
          yaw_err = wrap180(heading_state.yaw * 0.1 - yaw0)
        u = max(-HEADING_TRIM_MAX,
                min(HEADING_TRIM_MAX, HEADING_SIGN * HEADING_GAIN * yaw_err))
      else:
        yaw_err, u = 0.0, 0.0
      # LEG_ORDER (FL, FR, BL, BR): indices 0,2 are LEFT-side, 1,3 RIGHT-side
      trim = np.array([1.0 - u, 1.0 + u, 1.0 - u, 1.0 + u])

      action = policy(obs)
      for a, (unit, _, mask), sign, tr in zip(action, legs, LEG_SIGNS, trim):
        motor = le.MOTOR_LEFT if mask == 1 else le.MOTOR_RIGHT
        rad_s = float(a) * MAX_CRANK_SPEED * sign
        pct = min(MAX_SPEED_PCT, abs(rad_s) / math.radians(MAX_DEG_S) * 100.0) * tr
        pct = min(MAX_SPEED_PCT, max(0.0, pct))
        if pct < 3:
          unit.motor_stop(motor=motor)
        else:
          direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if rad_s >= 0
                       else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
          unit.motor_run(direction=direction, motor=motor, speed=int(pct))
      prev_action = action.copy()

      if DEBUG:
        print(f"\rclk {frac:.2f} | g_b {g_b[0]:+.2f}{g_b[1]:+.2f}{g_b[2]:+.2f} | "
              f"tilt {tilt_from_stand:3.0f} | yaw_err {yaw_err:+5.1f} | vel " +
              " ".join(f"{x:+4.1f}" for x in vel) + "   ", end="")

      phase_t += dt
      lag = time.time() - t0
      if lag < dt:
        time.sleep(dt - lag)
  except KeyboardInterrupt:
    pass
  finally:
    for unit, _, mask in legs:
      unit.motor_stop(motor=(le.MOTOR_LEFT if mask == 1 else le.MOTOR_RIGHT))
    front.disconnect()
    back.disconnect()
    print("\nstopped")


if __name__ == "__main__":
  main()
