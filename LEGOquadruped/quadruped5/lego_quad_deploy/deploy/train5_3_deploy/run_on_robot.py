"""Run the trained train5.3 policy on the REAL LEGO quadruped.

    python run_on_robot.py   (Python 3.14+ hardware env; pip install legoeducation numpy)

Needs policy_weights6.npz (from export_policy.py) and numpy_policy.py alongside.

IMPORTANT - this policy REQUIRES live IMU. train5.3 was trained with
imu_mode="live" (real gyro + gravity). Keep USE_LIVE_IMU = True; the frozen
zero-IMU mode is off-distribution for it (in sim it still walks forward
zeroed, but curves ~3x more). With live IMU it walks essentially STRAIGHT
in sim: +3.6 deg yaw drift and +1.04 m over 30 s (vs train5.1's -66 deg
curve - the drift fix worked).

EXPECTED GAIT - this policy is FRONT-HEAVY: the two FRONT cranks (FL, FR)
run pinned near max forward while the two HIND cranks (HL, HR) barely move
(mean action ~-0.19, mostly small oscillation). That is the trained
behavior, not a fault - so expect the front motors to spin steadily and the
back motors to mostly idle/twitch. It walks slowly (~0.03-0.05 m/s). If it
locomotes poorly on hardware because of this front-heavy pattern, the mesh-
model track (train6.x) is the path to a proper all-leg gait.

HEADING_HOLD stays on as a safety net, but sim drift is already small, so it
should barely intervene.

IMU + motor data come via notification CALLBACKS (the real API), cached and
read by the control loop. Units learned from device dumps:
  pitch/roll  = decidegrees (x0.1 -> deg)
  gyroscope   = decidegrees/s (x0.1 -> deg/s)
  accelerometer = milli-g; normalized -> gravity unit vector (body frame)
  MotorNotification.motorBitMask: 1 = LEFT, 2 = RIGHT
Ctrl+C stops the motors and exits.
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

COMMAND = 0.15
MAX_DEG_S = 700.0
MAX_SPEED_PCT = 40
LEG_SIGNS = np.array([-1.0, +1.0, -1.0, +1.0])   # (FL, FR, HL, HR)
IMU_UNIT = "front"

LEG_MAP = {
  "FL": ("front", "left"),
  "FR": ("front", "right"),
  "HL": ("back", "right"),
  "HR": ("back", "left"),
}
CONTROL_HZ = 5.0
MAX_CRANK_SPEED = 12.0
TILT_STOP_DEG = 75.0    # tilt from standing (via accelerometer, doesn't wrap)
TILT_STOP_HOLD = 5       # must exceed for this many consecutive loops to stop
USE_LIVE_IMU = True   # True: feed real accelerometer-gravity + gyro.
                      # train5.3 REQUIRES this (trained on live IMU); zeroed is
                      # off-distribution. Do not A/B this one to False.
GYRO_SIGNS = (1.0, 1.0, 1.0)

# --- heading hold: steer with differential leg speed to keep yaw constant ---
# Uses the FRONT unit's fused yaw (user-verified: constant yaw = straight).
HEADING_HOLD = True
HEADING_UNIT = "front"
HEADING_GAIN = 0.02    # differential fraction per degree of yaw error
HEADING_SIGN = +1.0    # flip to -1 if it steers the wrong way (amplifies drift)
HEADING_TRIM_MAX = 0.3 # cap the differential at +/-30%
DEBUG = True
# ------------------------------------------------

MASK = {"left": 1, "right": 2}
policy = NumpyPolicy("policy_weights6.npz")


class UnitState:
  """Caches the latest notification values for one Double Motor (thread-safe)."""
  def __init__(self):
    self.lock = threading.Lock()
    self.abspos = {1: 0.0, 2: 0.0}   # absolutePosition per mask, degrees
    self.speed = {1: 0.0, 2: 0.0}    # speed% per mask
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
  dm.device_notification_request(50)   # push updates every 50 ms
  print(f"{name} connected")
  return dm


def main():
  fstate, bstate = UnitState(), UnitState()
  front = connect(FRONT_CARD, "front", fstate)
  back = connect(BACK_CARD, "back", bstate)
  state = {"front": fstate, "back": bstate}
  front.imu_set_yaw_face(yaw_face = le.DEVICE_FACE_BACK)
  imu_state = state[IMU_UNIT]

  # (unit_obj, unit_state, mask) per leg, policy order FL, FR, HL, HR
  units = {"front": front, "back": back}
  legs = [(units[LEG_MAP[k][0]], state[LEG_MAP[k][0]], MASK[LEG_MAP[k][1]])
          for k in ("FL", "FR", "HL", "HR")]

  time.sleep(0.5)  # let the first notifications arrive

  input("\nrotate all four cranks so each FOOT is at its LOWEST point,\n"
        "then press Enter to anchor phase zero... ")
  with_lock = lambda st, m: st.abspos[m]
  ang0 = np.array([math.radians(with_lock(st, m)) for _, st, m in legs])
  print("phase anchored")

  # reference gravity + tilt while standing still
  time.sleep(0.3)
  with imu_state.lock:
    a0 = np.array(imu_state.accel, float)
    pitch0 = imu_state.pitch * 0.1
    roll0 = imu_state.roll * 0.1
  g_ref = a0 / (np.linalg.norm(a0) + 1e-9)

  # IMU->body alignment: rotation R0 sending measured standing gravity to
  # (0,0,-1), the value the policy saw when standing in training. Applied to
  # accel AND gyro. (Rotation about vertical stays unknown - gravity can't
  # see yaw - so gyro x/y may be mixed by a fixed yaw; usually tolerable.)
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
    yaw0 = heading_state.yaw * 0.1   # decidegrees -> degrees
  print(f"reference heading yaw0 = {yaw0:+.0f} deg (point the robot the way you want it to go)")

  def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0

  prev_action = np.zeros(4)
  tilt_count = 0
  dt = 1.0 / CONTROL_HZ
  print(f"running (USE_LIVE_IMU={USE_LIVE_IMU}) - Ctrl+C to stop")
  try:
    while True:
      t0 = time.time()

      # ---- read cached sensor state ----
      angs, spds = [], []
      for _, st, m in legs:
        with st.lock:
          angs.append(st.abspos[m]); spds.append(st.speed[m])
      with imu_state.lock:
        pitch = imu_state.pitch * 0.1 - pitch0
        roll = imu_state.roll * 0.1 - roll0
        gyro_raw = np.array(imu_state.gyro, float) * 0.1   # deg/s
        accel = np.array(imu_state.accel, float)

      # tilt safety: angle between current gravity and standing gravity.
      # Uses the accelerometer (no wrap), needs several bad reads in a row.
      g_now = accel / (np.linalg.norm(accel) + 1e-9)
      tilt_from_stand = math.degrees(
        math.acos(max(-1.0, min(1.0, float(g_now @ g_ref)))))
      if tilt_from_stand > TILT_STOP_DEG:
        tilt_count += 1
        if tilt_count >= TILT_STOP_HOLD:
          print(f"\ntilt limit ({tilt_from_stand:.0f} deg from standing"
                f", sustained) - stopping")
          break
      else:
        tilt_count = 0

      # ---- build observation ----
      if USE_LIVE_IMU:
        g_b = R0 @ (accel / (np.linalg.norm(accel) + 1e-9))
        gyro = R0 @ (np.radians(gyro_raw) * np.array(GYRO_SIGNS))
      else:
        g_b = np.array([0.0, 0.0, -1.0])   # frozen-IMU mode
        gyro = np.zeros(3)

      ang = (np.radians(angs) - ang0) * LEG_SIGNS
      vel = np.radians(np.array(spds) / 100.0 * MAX_DEG_S) * LEG_SIGNS

      obs = np.concatenate([
        np.zeros(3),          # base lin vel: trained blind
        gyro, g_b,
        np.sin(ang), np.cos(ang),
        vel / 10.0,
        prev_action, [COMMAND],
      ])

      # ---- heading hold: differential trim from yaw drift ----
      if HEADING_HOLD:
        with heading_state.lock:
          yaw_err = wrap180(heading_state.yaw * 0.1 - yaw0)
        u = max(-HEADING_TRIM_MAX,
                min(HEADING_TRIM_MAX, HEADING_SIGN * HEADING_GAIN * yaw_err))
      else:
        yaw_err, u = 0.0, 0.0
      # legs order (FL, FR, HL, HR): indices 0,2 are LEFT, 1,3 are RIGHT
      trim = np.array([1.0 - u, 1.0 + u, 1.0 - u, 1.0 + u])

      # ---- policy -> motor commands ----
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
        print(f"\rg_b {g_b[0]:+.2f}{g_b[1]:+.2f}{g_b[2]:+.2f} | "
              f"tilt {tilt_from_stand:3.0f} | yaw_err {yaw_err:+5.1f} deg | vel " +
              " ".join(f"{v:+4.1f}" for v in vel) + "   ", end="")

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
