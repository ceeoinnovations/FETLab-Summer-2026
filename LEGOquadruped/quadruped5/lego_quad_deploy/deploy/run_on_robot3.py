"""Run the trained policy on the REAL LEGO quadruped.

    python run_on_robot.py     (Python 3.14+, pip install legoeducation numpy)

Needs policy_weights2.npz (from export_policy.py) and numpy_policy.py
in the same folder. Fill in the connection cards + calibration below.
Ctrl+C stops the motors and exits.

Observation layout matches training (lego_env_deploy.py):
  [ lin_vel(=0,0,0) | gyro(rad/s) | gravity_dir | sin(crank) | cos(crank)
    | crank_vel/10 | prev_action | command ]
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import legoeducation as le
from numpy_policy import NumpyPolicy

# ---------------- fill these in ----------------
FRONT_CARD = dict(card_color=le.LEGO_COLOR_RED, card_serial="1779")
BACK_CARD = dict(card_color=le.LEGO_COLOR_PURPLE, card_serial="6040")

COMMAND = 0.15        # target forward speed (m/s); start LOW
MAX_DEG_S = 700.0     # measured deg/s at 100% speed (calibrate_motors.py)
MAX_SPEED_PCT = 40    # hard cap on motor speed%, for safe/slow testing (100 = off)
# per-leg sign so +action = forward gait, order (FL, FR, HL, HR):
LEG_SIGNS = np.array([-1.0, +1.0, -1.0, +1.0])
IMU_UNIT = "back"     # which double-motor's IMU is rigidly on the body

# Which (unit, port) drives each PHYSICAL leg. Run the "identify legs"
# step in calibrate_motors.py and copy its printout here verbatim.
LEG_MAP = {
  "FL": ("front", "left"),
  "FR": ("front", "right"),
  "HL": ("back", "right"),
  "HR": ("back", "left"),
}
CONTROL_HZ = 5.0   # matched to measured BLE latency + retrained policy
MAX_CRANK_SPEED = 30.0  # rad/s, must match training
TILT_STOP_DEG = 60.0    # safety: stop if tilted past this FROM the start pose
GYRO_SIGNS = (1.0, 1.0, 1.0)  # flip entries if the robot reacts backwards to rotation
# IMU raw-unit scales: raw * scale = degrees (or deg/s).
# Your standing reading was pitch=+848 raw -> with 0.01 that is 8.5 deg. VERIFY:
# tilt the robot ~45 deg by hand and check the debug tilt reads ~45.
PITCH_ROLL_SCALE = 0.01
GYRO_SCALE = 0.01
DEBUG = True          # print live obs sanity numbers (turn off once happy)
# ------------------------------------------------

policy = NumpyPolicy("policy_weights2.npz")


def connect(card, name):
  dm = le.DoubleMotor()
  dm.connect(**card)
  if not dm.connected:
    raise SystemExit(f"could not connect to {name} double motor")
  print(f"{name} connected")
  return dm


def main():
  front = connect(FRONT_CARD, "front")
  back = connect(BACK_CARD, "back")
  imu = (back if IMU_UNIT == "back" else front).imu_device

  # (unit, motor_const) per leg in policy order FL, FR, HL, HR,
  # resolved from the user-verified LEG_MAP.
  units = {"front": front, "back": back}
  ports = {"left": le.MOTOR_LEFT, "right": le.MOTOR_RIGHT}
  legs = [(units[LEG_MAP[k][0]], ports[LEG_MAP[k][1]])
          for k in ("FL", "FR", "HL", "HR")]

  # --- crank phase anchoring ---
  # The encoders' zero is arbitrary per run; the policy needs angles measured
  # from the sim's zero pose (crank at bottom, foot at its LOWEST point).
  input("\nrotate all four cranks so each FOOT is at its LOWEST point,\n"
        "then press Enter to anchor phase zero... ")
  ang0 = np.array([math.radians(u.motor[m].absolutePosition) for u, m in legs])
  print("phase anchored")

  # --- reference orientation: whatever the IMU reads while standing ---
  # The IMU unit is mounted flipped/rotated, so its raw pitch/roll are not
  # zero at rest. All tilt is measured RELATIVE to this startup reading.
  time.sleep(0.5)
  pitch0 = math.radians(imu.pitch * PITCH_ROLL_SCALE)
  roll0 = math.radians(imu.roll * PITCH_ROLL_SCALE)
  print(f"reference orientation: pitch {math.degrees(pitch0):+.0f} deg, "
        f"roll {math.degrees(roll0):+.0f} deg  (stand the robot still at start)")

  def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

  prev_action = np.zeros(4)
  dt = 1.0 / CONTROL_HZ
  pool = ThreadPoolExecutor(max_workers=2)
  print("running - Ctrl+C to stop")
  try:
    while True:
      t0 = time.time()

      # ---- observations (front/back read in parallel: BLE is slow) ----
      def read_front():
        return [(front.motor[m].absolutePosition, front.motor[m].speed)
                for u, m in legs[:2]]
      def read_back():
        vals = [(back.motor[m].absolutePosition, back.motor[m].speed)
                for u, m in legs[2:]]
        return vals, imu.pitch, imu.roll, imu.gyroscopeX, imu.gyroscopeY, imu.gyroscopeZ
      f_future = pool.submit(read_front)
      b_vals, p_raw, r_raw, gx, gy, gz = pool.submit(read_back).result()
      f_vals = f_future.result()
      m_pos = np.array([v[0] for v in f_vals + b_vals])
      m_spd = np.array([v[1] for v in f_vals + b_vals])

      # tilt relative to the startup (standing) orientation
      pitch = wrap(math.radians(p_raw * PITCH_ROLL_SCALE) - pitch0)
      roll = wrap(math.radians(r_raw * PITCH_ROLL_SCALE) - roll0)
      if abs(math.degrees(pitch)) > TILT_STOP_DEG or \
         abs(math.degrees(roll)) > TILT_STOP_DEG:
        print(f"\ntilt limit ({math.degrees(pitch):+.0f}/"
              f"{math.degrees(roll):+.0f} deg from start) - stopping")
        break

      gyro = np.radians(np.array([gx, gy, gz]) * GYRO_SCALE)
      gyro = gyro * np.array(GYRO_SIGNS)
      # gravity direction in body frame from tilt-from-standing
      g_b = np.array([math.sin(pitch),
                      -math.sin(roll) * math.cos(pitch),
                      -math.cos(roll) * math.cos(pitch)])

      ang = (np.radians(m_pos) - ang0) * LEG_SIGNS
      vel = np.radians(m_spd / 100.0 * MAX_DEG_S) * LEG_SIGNS

      obs = np.concatenate([
        np.zeros(3),          # base lin vel: unmeasurable, trained blind
        gyro, g_b,
        np.sin(ang), np.cos(ang),
        vel / 10.0,
        prev_action, [COMMAND],
      ])

      # ---- policy -> motor commands ----
      action = policy(obs)
      for a, (unit, motor), sign in zip(action, legs, LEG_SIGNS):
        rad_s = float(a) * MAX_CRANK_SPEED * sign
        pct = min(MAX_SPEED_PCT, abs(rad_s) / math.radians(MAX_DEG_S) * 100.0)
        if pct < 3:
          unit.motor_stop(motor=motor)
        else:
          direction = (le.MOTOR_MOVE_DIRECTION_CLOCKWISE if rad_s >= 0
                       else le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE)
          unit.motor_run(direction=direction, motor=motor, speed=int(pct))
      prev_action = action.copy()

      if DEBUG:
        print(f"\rvel(rad/s) " +
              " ".join(f"{v:+5.1f}" for v in vel) +
              f" | tilt {math.degrees(pitch):+4.0f}/{math.degrees(roll):+4.0f} deg  ",
              end="")

      lag = time.time() - t0
      if lag < dt:
        time.sleep(dt - lag)
      else:
        print(f"\rloop overrun {lag*1000:.0f} ms ", end="")
  except KeyboardInterrupt:
    pass
  finally:
    for unit, motor in legs:
      unit.motor_stop(motor=motor)
    front.disconnect()
    back.disconnect()
    print("\nstopped")


if __name__ == "__main__":
  main()