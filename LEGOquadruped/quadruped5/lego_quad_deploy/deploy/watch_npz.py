"""Watch the TRAINED policy (.npz) drive YOUR REAL LEGO MODEL (meshes+linkage).

    uv run python play_on_lego.py policy_weights4.npz zero
    uv run python play_on_lego.py policy_weights3.npz live

Needs lego_walk_scene.xml + assets/ (your STLs) in the same folder,
plus numpy_policy.py. This is the sim-to-sim dress rehearsal: the policy
trained on the abstract crank model; here it drives the faithful four-bar
linkage model instead. Close the window to stop.
"""

import re
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from numpy_policy import NumpyPolicy

POLICY = sys.argv[1] if len(sys.argv) > 1 else "policy_weights.npz"
IMU_MODE = sys.argv[2] if len(sys.argv) > 2 else "zero"
assert IMU_MODE in ("zero", "live")

COMMAND = 0.15
MAX_CRANK_SPEED = 12.0
# gear-servo strength: raise FORCERANGE if any gear stalls (torque-saturated)
KV = 1.0
FORCERANGE = 1.5
FRICTIONLOSS = 0.02   # export default 0.1 per joint is unrealistically draggy
CONTROL_DT = 0.2                     # 5 Hz, matches deploy training
# crank direction per leg, policy order (FL, FR, HL, HR) - same as hardware
LEG_SIGNS = np.array([-1.0, +1.0, -1.0, +1.0])

policy = NumpyPolicy(POLICY)

# load the faithful scene; convert the 4 gear position-servos to velocity
xml = open("lego_walk_scene.xml").read()
xml = re.sub(
  r'<position class="([^"]*)" name="(\w+Gear)" joint="(\w+)"/>',
  rf'<velocity class="\1" name="\2" joint="\3" kv="{KV}" '
  rf'forcerange="-{FORCERANGE} {FORCERANGE}" ctrlrange="-12 12"/>',
  xml)
xml = xml.replace('frictionloss="0.1"', f'frictionloss="{FRICTIONLOSS}"')

# Disable robot SELF-collision (mesh parts at true LEGO clearances jam the
# solver - the stuck back gears) while keeping robot-vs-floor contact:
#   robot collision geoms: contype=1, conaffinity=0  (can't hit each other)
#   floor:                 contype=0, conaffinity=1  (feet-floor still collides)
xml = re.sub(r'(<default class="collision">\s*<geom) (group="3")',
             r'\1 contype="1" conaffinity="0" \2', xml)
xml = re.sub(r'(<geom name="floor" type="plane")',
             r'\1 contype="0" conaffinity="1"', xml)
# stiffen the loop-closure welds for continuous crank rotation
xml = xml.replace('<weld body1=',
                  '<weld solref="0.002 1" solimp="0.95 0.99 0.001" body1=')

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

GEARS = ("frontLeftGear", "frontRightGear", "backLeftGear", "backRightGear")
ACT = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in GEARS]
QPOS = [model.jnt_qposadr[model.joint(n).id] for n in GEARS]
QVEL = [model.jnt_dofadr[model.joint(n).id] for n in GEARS]
N_SUB = int(round(CONTROL_DT / model.opt.timestep))

# settle onto the feet, then anchor gear angles at the standing pose
for _ in range(400):
  mujoco.mj_step(model, data)
ang0 = np.array([data.qpos[i] for i in QPOS])
print(f"contacts at standing: {data.ncon} "
      "(should be small - just feet on floor; dozens = self-collision jam)")

prev_action = np.zeros(4)


def get_obs():
  quat = data.qpos[3:7]
  R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
  if IMU_MODE == "live":
    gyro = R.T @ data.qvel[3:6]
    g_b = R.T @ np.array([0.0, 0.0, -1.0])
  else:
    gyro = np.zeros(3)
    g_b = np.array([0.0, 0.0, -1.0])
  ang = (np.array([data.qpos[i] for i in QPOS]) - ang0) * LEG_SIGNS
  vel = np.array([data.qvel[i] for i in QVEL]) * LEG_SIGNS
  return np.concatenate([np.zeros(3), gyro, g_b,
                         np.sin(ang), np.cos(ang), vel / 10.0,
                         prev_action, [COMMAND]])


print(f"driving YOUR model with {POLICY} (imu_mode={IMU_MODE})")
with mujoco.viewer.launch_passive(model, data) as viewer:
  viewer.cam.distance = 0.8
  viewer.cam.elevation = -18
  x_prev, t_prev = data.qpos[0], data.time
  while viewer.is_running():
    t0 = time.time()
    action = np.clip(policy(get_obs()), -1, 1)
    for i, a in enumerate(ACT):
      data.ctrl[a] = LEG_SIGNS[i] * action[i] * MAX_CRANK_SPEED
    for _ in range(N_SUB):
      mujoco.mj_step(model, data)
    prev_action = action.copy()
    viewer.cam.lookat[:] = data.qpos[0:3]
    viewer.sync()
    spd = (data.qpos[0] - x_prev) / max(data.time - t_prev, 1e-6)
    x_prev, t_prev = data.qpos[0], data.time
    print(f"\rfwd {spd:+.2f} m/s (cmd {COMMAND})   ", end="")
    dt = CONTROL_DT - (time.time() - t0)
    if dt > 0:
      time.sleep(dt)