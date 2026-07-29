# LEGO quadruped - hardware deployment

Pipeline: retrain (hardware-matched) -> export -> calibrate -> run.

## Why retrain first
Your current policy observes base linear velocity, which no sensor on the
robot can measure, and it was trained at 50 Hz control, which Bluetooth
cannot sustain. `train_deploy.py` retrains with lin-vel blinded, 20 Hz
control, and extra randomization (motor gain error, gyro noise) so the
policy tolerates imperfect calibration. Same speed as before (~20-40 min).

## Steps

1. TRAIN the deployment policy (in your uv project, next to lego_env.py
   and lego_quad_cpu.xml):

       uv run python train_deploy.py

   Sanity-watch it: watch_policy.py works if you point it at
   lego_quad_deploy_ppo.zip after changing its env import to
   LegoQuadDeployEnv (or just trust ep_rew_mean).

2. EXPORT to NumPy (no torch needed on the robot side):

       uv run python export_policy.py lego_quad_deploy_ppo.zip

   -> policy_weights.npz  (verified to match SB3 to ~1e-7)

3. HARDWARE ENV: the LEGO API needs Python >= 3.14. Make a separate env:

       py -3.14 -m venv lego-env
       lego-env\Scripts\activate
       pip install legoeducation numpy

   Copy into one folder: run_on_robot.py, calibrate_motors.py,
   numpy_policy.py, policy_weights.npz.

4. CALIBRATE:  python calibrate_motors.py
   - fill in your two Connection Cards (color + serial) first
   - gives you MAX_DEG_S per leg and the LEG_SIGNS (+1/-1) entries

5. RUN:        python run_on_robot.py
   - fill in cards, MAX_DEG_S, LEG_SIGNS, IMU_UNIT
   - start with COMMAND = 0.10-0.15 m/s, robot on a rug/carpet
   - Ctrl+C stops all motors; it also auto-stops past 60 deg tilt

## Expectations (honest)
- First runs usually look worse than sim: BLE latency, motor deadband,
  foot slip, and the four-bar-vs-circle gap all bite. That is normal.
- Best knobs, in order: lower COMMAND; verify LEG_SIGNS (a single wrong
  sign ruins everything); re-measure MAX_DEG_S; try CONTROL_HZ 10-15 if
  you see "loop overrun" messages.
- If it walks but drifts or stumbles, more training with wider
  randomization in lego_env_deploy.py is the lever.
- The IMU yaw/pitch/roll axes depend on how the unit is mounted; if the
  robot reacts wrongly to tilting, the g_b / gyro axis signs in
  run_on_robot.py may need flipping to match your mounting.
