"""Watch the train13 policy walk in the live MuJoCo viewer.

    uv run python watch_train13.py            # cold-start (walk from rest)
    uv run python watch_train13.py 0.04        # set command speed (m/s)

Needs a DISPLAY (won't work on the headless cluster - there, render an mp4 with
analyze_final.py instead). Needs alongside: lego_env13.py, lego_quad_mesh.xml,
../assets/, numpy_policy.py, policy_weights13.npz. Mouse orbits/zooms; close the
window or Ctrl+C to stop.
"""
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

import lego_env13
from lego_env13 import LegoQuadEnv
sys.path.insert(0, "..")
from numpy_policy import NumpyPolicy

COMMAND = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
policy = NumpyPolicy("policy_weights13.npz")

lego_env13.RSI_PROB = 0.0          # cold-start so you see it initiate from rest
env = LegoQuadEnv(control_dt=0.2, episode_s=1e9, randomize=False)
obs, _ = env.reset(seed=0)
env.cmd = COMMAND
env.max_steps = 10**9

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
  viewer.cam.distance = 1.1
  viewer.cam.elevation = -20
  print(f"watching train13 @ cmd {COMMAND} m/s (close window to stop)")
  while viewer.is_running():
    t0 = time.time()
    obs[0:3] = 0.0                 # blind lin-vel, as deployed
    obs, r, term, trunc, info = env.step(policy(obs))
    if term:                       # fell -> reset and keep going
      obs, _ = env.reset(); env.cmd = COMMAND
    viewer.cam.lookat[:] = env.data.qpos[0:3]
    viewer.sync()
    dt = 0.2 - (time.time() - t0)
    if dt > 0:
      time.sleep(dt)
