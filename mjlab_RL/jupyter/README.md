# jupyter/

Sim-to-real deployment: takes a policy trained in `hpc/` (mjlab + PPO) and runs it on the physical UR3e arm, then plots the logged rollout.

## Contents

- **`myur3e.py`** — `MyUR3e`, a ROS2 (`rclpy`) node for controlling the real UR3e arm: inverse kinematics, trajectory planning, and joint/force-torque state subscriptions. Written by Aengus Kennedy and Liam Campbell (Center for Engineering Education and Outreach, Summer 2024). This is the hardware interface the notebooks below drive.
- **`RL_implementation.ipynb`** — loads a trained policy checkpoint (from `hpc/`'s training runs) and runs it on the real arm via `myur3e.py`. Step duration and action scaling are slowed down from training-time values for safety on real hardware.
- **`plot_rollout_log (1).ipynb`** — reads the CSV rollout log written during a real-arm run (`timestamp`, `pos_<joint>` x6, `vel_<joint>` x6) and plots joint position and velocity vs. time, for comparing real-robot behavior against the simulated policy.
- **`2026-07-23_14-02-34.pt`** — an example trained checkpoint used with the notebooks above.

## Requirements

- ROS2 (the specific distro your UR3e driver targets) with `rclpy` and the UR3e's ROS2 driver running.
- A physical UR3e arm, reachable over the network/USB as configured by the ROS2 driver.
- Python packages: `numpy`, `scipy`, `torch` (to load the `.pt` checkpoint), plus whatever `myur.ik_solver` / `myur.trajectory_planner` depend on.
- Jupyter, obviously.

## Usage

1. Train a policy in `hpc/` and copy its checkpoint here (or point directly at the `logs/rsl_rl/...` path).
2. Start the UR3e ROS2 driver so `myur3e.py` can connect to the real arm.
3. Run `RL_implementation.ipynb`, pointing it at your checkpoint. **Double-check step duration and action scaling before running on real hardware** — training-time values are not safe defaults for the physical arm.
4. It writes a rollout log CSV; point `plot_rollout_log (1).ipynb`'s `LOG_PATH` at that file and run all cells to visualize joint position/velocity over time.

⚠️ Running a policy on a real robot arm can cause unexpected or unsafe motion, especially early checkpoints or unfamiliar tasks. Keep a hand near the e-stop and start with conservative action scaling.
