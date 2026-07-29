"""Watch YOUR LEGO quadruped (real linkage + meshes) walk in the MuJoCo viewer.

Put this next to lego_walk_scene.xml and your assets/ folder, then:
    uv run python walk_lego.py

Tuning knobs (real foot meshes may need slightly different values):
  STEP_FREQ - crank speed; lower if it stumbles or pitches too much
  LEAN_DEG  - forward lean at start; raise if it still falls backward
  SIGNS     - flip all four if it walks backward
"""
import time
import numpy as np
import mujoco
import mujoco.viewer

STEP_FREQ = 1.0          # crank revolutions per second
RAMP_S    = 2.0          # soft start: reach full speed over this many seconds
LEAN_DEG  = 3.0          # forward lean added at start (counters backward falls)
SIGNS = (-1, 1, 1, -1)   # crank directions for (FL, BR, FR, BL)
FWD = np.array([-0.97, -0.25])  # world walk direction for this pose/sign set

model = mujoco.MjModel.from_xml_path("lego_walk_scene.xml")
data = mujoco.MjData(model)

FL, BR, FR, BL = (
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
    for n in ("frontLeftGear", "backRightGear", "frontRightGear", "backLeftGear")
)

# apply the forward lean to the starting orientation
axis = np.array([-FWD[1], FWD[0], 0.0])
q_lean = np.zeros(4)
mujoco.mju_axisAngle2Quat(q_lean, axis, np.radians(LEAN_DEG))
q_new = np.zeros(4)
mujoco.mju_mulQuat(q_new, q_lean, data.qpos[3:7].copy())
data.qpos[3:7] = q_new

# settle onto the feet
for _ in range(400):
    mujoco.mj_step(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 0.8
    viewer.cam.elevation = -20
    phase = 0.0
    while viewer.is_running():
        t0 = time.time()
        # frequency ramps up smoothly, phase integrates so there is no jump
        freq = STEP_FREQ * min(1.0, data.time / RAMP_S)
        phase += 2.0 * np.pi * freq * model.opt.timestep
        data.ctrl[FL] = SIGNS[0] * phase
        data.ctrl[BR] = SIGNS[1] * phase
        data.ctrl[FR] = SIGNS[2] * (phase + np.pi)
        data.ctrl[BL] = SIGNS[3] * (phase + np.pi)
        mujoco.mj_step(model, data)
        viewer.cam.lookat[:] = data.qpos[0:3]
        viewer.sync()
        dt = model.opt.timestep - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)