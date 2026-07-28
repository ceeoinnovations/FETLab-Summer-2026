"""Run the trained train19.1 policy on the REAL LEGO quadruped.

    python run_on_robot19_1.py   (hardware env; pip install legoeducation numpy)

Copy these three files together onto the hardware machine:
    run_on_robot19_1.py, numpy_policy.py, policy_weights19_1.npz

train19.1 = the FLEXIBLE clock-free gait-coordination variant (barrier_adv_coef
=0.6) with RECOVERY training (broken-phase RSI seeds + mid-episode phase kicks +
heavier latency randomization). In SIM it walks DEAD STRAIGHT (~4 deg veer) and
recovers its heading from errors as large as 120 deg - the best steering of the
family - but it leaned all the way into FRONT drive (back cranks barely turn,
~front-only) and is slower (~0.018 m/s). Deploy to check hardware behaviour and,
especially, whether its strong steering recovery carries over to the real robot.
Same crank convention and 30-dim heading-in-obs as run_on_robot15/17.1/18/19.

Notes:

1. OBS IS 30-dim: v13's 28 + sin/cos of the heading error (yaw - yaw0). The
   phase clock (CLOCK_T=0.72 s) advances on wall-clock time from start.

2. HEADING ERROR = the front IMU's fused yaw drift from the start heading. Point
   the robot the way you want to go at startup (that sets yaw0), and the policy
   holds it. Yaw is an orientation, so where the IMU sits on the body doesn't
   matter - only the axis (vertical) and SIGN do.

3. HEADING_ERR_SIGN is the one thing to verify: sim yaw increases CCW; if the
   IMU increases the other way, the policy steers the WRONG way and drift grows.
   VERIFY SLOWLY FIRST - if it curves worse when running, set HEADING_ERR_SIGN=-1.
   (This is the prime suspect for the real robot "turning so much it can't
   readjust": in sim this policy recovers from 120 deg, so a runaway turn on
   hardware points at a wrong sign / noisy IMU yaw rather than the policy.)

4. DIRECTION: walks your original forward. If backward, flip all LEG_SIGNS.
   Sanity-check LEG_MAP/LEG_SIGNS with scripted_trot_test.py (one leg fights ->
   that leg's back-unit L/R swap).

5. FRONT-ONLY GAIT: unlike train18, this policy barely drives the back cranks by
   design, so lightly-moving back legs are expected here, not a fault.

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

COMMAND = 0.03          # commanded forward speed (m/s). train19.1 range 0.01-0.04.
CLOCK_T = 0.72          # gait-clock period (s) - MUST match training
MAX_DEG_S = 700.0
MAX_SPEED_PCT = 40      # start LOW (~15) the first time, then raise once verified
# per-crank sign, policy/obs order (FL, FR, BL, BR) = mesh GEAR_JOINTS order.
# The policy's DOMINANT crank directions are [-,+,-,+] (= RSI_SIGNS); LEG_SIGNS
# maps the policy action -> motor direction (CW if action*sign >= 0). With +1s
# the trot passes straight through. NOTE: an [-1,+1,-1,+1] LEG_SIGNS would cancel
# the policy's own [-,+,-,+] into all-one-direction = the flail bug.
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

# --- v15: the POLICY steers itself from the heading error in its obs ---
# We feed sin/cos(HEADING_ERR_SIGN * (imu_yaw - yaw0)) into the last 2 obs dims.
# HEADING_ERR_SIGN: sim yaw increases CCW; if the IMU increases the OTHER way the
# policy will steer the WRONG way (drift grows) - flip this to -1 then. Verify on
# a slow first run: if it curves worse when running, flip the sign.
HEADING_ERR_SIGN = -1.0
HEADING_UNIT = "front"   # IMU unit whose fused yaw is the heading (center-ish)

# External heading-hold trim is now OFF - the policy handles steering. Left here
# as an optional extra correction if the policy alone still curves.
HEADING_HOLD = False
HEADING_GAIN = 0.02
HEADING_SIGN = +1.0
HEADING_TRIM_MAX = 0.3
DEBUG = True

# --- anti-stall: pre-roll into a trot + phase-lock watchdog (front AND back) ---
# The trained gait needs the two FRONT cranks ANTIPHASE and the two BACK cranks
# ANTIPHASE - they counter-rotate (the policy's dominant action is [-,+,-,+] =
# RSI_SIGNS). If a pair locks into CO-rotation (same direction, same phase) both
# feet do the same thing at once and the robot bobs without translating.
# RSI_SIGNS is the trot drive the policy trained on; we reuse it to (a) PRE-ROLL
# the cranks into a moving trot at startup instead of launching all-feet-down
# (all in phase = the stall configuration), and (b) NUDGE a locked pair apart.
# (train19.1 barely drives the back cranks, so the back-pair check will seldom
# fire - it needs real motion in both cranks to trigger.)
RSI_SIGNS = np.array([-1.0, 1.0, -1.0, 1.0])
PREROLL_ENABLE = True
PREROLL_CYCLES = 2.0     # crank revolutions to spin up before handing to policy
WATCHDOG_ENABLE = True
CO_ROT_DELTA = 0.3       # rad/step: min crank motion to count (normal ~1.7)
CO_ROT_STEPS = 5         # sustained co-rotating steps (~1 s at 5 Hz) => phase-lock
NUDGE_STEPS = 3          # open-loop trot-drive steps to break the lock

# --- clock-from-cranks: slave the gait clock to ACTUAL crank motion ---
# The gait clock in the obs normally free-runs on wall-clock time (phase_t += dt).
# In sim the velocity-controlled cranks track that exactly; on hardware they lag
# (BLE latency, friction, load), so the fixed clock drifts against the slower
# real cadence and the two BEAT: the robot walks when clock ~= crank phase and
# stalls in place when the clock runs ~half a cycle ahead. Advancing the clock by
# the MEASURED crank progress instead keeps it locked to the cranks -> no beat.
# (On hardware this policy's stalls show wd climbing only to ~3 then back to 0 -
# a brief co-rotation it self-recovers - with the beat underneath.)
CLOCK_FROM_CRANKS = True
# The policy trained on a clock that advanced ~dt EVERY step and never froze, so
# clamp the crank-slaved advance to [MIN,MAX]*dt: it slows to track lagging cranks
# (kills the beat) but never FREEZES or LURCHES - a stopped/jumping clock is
# off-distribution and makes the policy command erratic full-speed bursts.
CLOCK_ADV_MIN = 0.4
CLOCK_ADV_MAX = 1.0
# ------------------------------------------------

MASK = {"left": 1, "right": 2}
policy = NumpyPolicy("policy_weights19_1.npz")


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


def _wrap_pi(a):
  return (a + math.pi) % (2.0 * math.pi) - math.pi


def drive_motors(legs, action, trim):
  """Send one crank-velocity action vector to the motors. Shared by the policy
  loop, the startup pre-roll, and the anti-stall nudge so they map actions to
  motor commands identically."""
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
  prev_ang = None
  tilt_count = 0
  dt = 1.0 / CONTROL_HZ
  phase_t = 0.0          # gait clock; advances dt each control step (as trained)
  omega_frac = (2.0 * math.pi / CLOCK_T) / MAX_CRANK_SPEED   # action -> rad_s = omega
  omega = 2.0 * math.pi / CLOCK_T          # nominal crank speed (rad/s)

  # pre-roll the cranks into a moving trot (open-loop RSI pattern) before handing
  # to the policy, so we launch from the mid-trot state the policy trained on
  # instead of all-cranks-in-phase (every foot down) - the degenerate phase-
  # locked configuration the robot stalls in.
  if PREROLL_ENABLE:
    n_pre = int(round(PREROLL_CYCLES * CLOCK_T / dt))
    print(f"pre-rolling {PREROLL_CYCLES:g} crank cycles into a trot...")
    for _ in range(n_pre):
      drive_motors(legs, RSI_SIGNS * omega_frac, np.ones(4))
      phase_t += dt
      time.sleep(dt)

  co_rot, nudge_left = 0, 0
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
      # per-crank angle change since last step (drives the watchdog + crank clock)
      d_crank = None if prev_ang is None else _wrap_pi(ang - prev_ang)

      # phase clock (concept 1): sin/cos of the gait phase
      frac = (phase_t / CLOCK_T) % 1.0
      p = 2.0 * math.pi * frac

      # v15 heading error: sin/cos of the IMU yaw drift from the start heading.
      # The policy uses this to steer. Sign must match sim (flip HEADING_ERR_SIGN
      # if it curves worse when running).
      with heading_state.lock:
        yaw_err_deg = wrap180(heading_state.yaw * 0.1 - yaw0)
      he = math.radians(HEADING_ERR_SIGN * yaw_err_deg)

      obs = np.concatenate([
        np.zeros(3),          # base lin vel: trained blind
        gyro, g_b,
        np.sin(ang), np.cos(ang),
        vel / 10.0,
        prev_action, [COMMAND],
        [math.sin(p), math.cos(p)],   # phase clock
        [math.sin(he), math.cos(he)],   # 30-dim: heading error (v15)
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

      # anti-stall watchdog: the two FRONT cranks (idx 0,1) should COUNTER-rotate,
      # and so should the two BACK cranks (idx 2,3). If EITHER pair's crank angles
      # advance the SAME way (co-rotate) at speed for a sustained spell, that pair
      # has phase-locked - inject a short open-loop trot drive (RSI pattern) to
      # re-seed all four legs, then hand control back. Angle deltas are robust
      # whether or not the motor speed notification is signed.
      if WATCHDOG_ENABLE and d_crank is not None:
        if nudge_left > 0:
          action = RSI_SIGNS * omega_frac
          nudge_left -= 1
        else:
          d = d_crank
          front_sync = d[0] * d[1] > 0.0 and min(abs(d[0]), abs(d[1])) > CO_ROT_DELTA
          back_sync = d[2] * d[3] > 0.0 and min(abs(d[2]), abs(d[3])) > CO_ROT_DELTA
          co_rot = co_rot + 1 if (front_sync or back_sync) else 0
          if co_rot >= CO_ROT_STEPS:
            nudge_left, co_rot = NUDGE_STEPS, 0
      prev_ang = ang.copy()

      drive_motors(legs, action, trim)
      prev_action = action.copy()

      if DEBUG:
        # yaw_err_deg is the live IMU drift the POLICY sees (yaw - yaw0); yaw_err
        # is the heading-hold trim, which is 0 when HEADING_HOLD is off. wd = the
        # watchdog state: the co-rotation counter, or NUDGE while breaking a lock.
        wd = "NUDGE" if nudge_left > 0 else str(co_rot)
        print(f"\rclk {frac:.2f} | g_b {g_b[0]:+.2f}{g_b[1]:+.2f}{g_b[2]:+.2f} | "
              f"tilt {tilt_from_stand:3.0f} | yaw_err {yaw_err_deg:+6.1f} | "
              f"wd {wd:>5} | vel " +
              " ".join(f"{x:+4.1f}" for x in vel) + "   ", end="")

      # advance the gait clock: by measured crank progress (locked to the real
      # cadence, no beat) or, if disabled, by wall-clock dt as trained.
      if CLOCK_FROM_CRANKS and d_crank is not None:
        phase_t += float(np.mean(np.abs(d_crank))) / omega
      else:
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
