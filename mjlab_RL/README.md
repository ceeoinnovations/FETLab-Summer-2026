# Summer RL + Robot Arm Project

Summer research project combining reinforcement learning for robot control (in [mjlab](https://github.com/mujocolab/mjlab), a MuJoCo-based RL framework) with deploying trained policies on a real UR3e robot arm. The RL tasks were trained on an HPC SLURM cluster and prototyped locally before submission.

## How the pieces fit together

1. **`local/`** — develop and sanity-check a task on a laptop/workstation (fast iteration, small `num_envs`, visual MuJoCo viewer).
2. **`hpc/`** — once a task works locally, train it for real on the cluster with many parallel envs and GPU acceleration.
3. **`jupyter/`** — take a trained checkpoint from `hpc/` and deploy it on the physical UR3e arm, then plot the logged rollout to compare sim vs. real behavior.

See each folder's own README for details:
- [`local/README.md`](local/README.md)
- [`hpc/README.md`](hpc/README.md)
- [`jupyter/README.md`](jupyter/README.md)

## Tasks in this repo

- **UR3e Reach** (`ur3e_reach`) — a fixed-base UR3e arm learns to reach a target end-effector position.
- **Lego Crawler / "Silly Walk"** (`silly_walk`) — a free-base LEGO-built crawler robot learns to walk forward at a commanded speed.
- **`template`** (in `local/`) — a blank-slate skeleton for scaffolding a new task; not runnable as-is.

## Setup

This project depends on [mjlab](https://github.com/mujocolab/mjlab) and uses [`uv`](https://docs.astral.sh/uv/) for Python dependency management.

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# from wherever you keep your mjlab-based project (e.g. local/my_mjlab_project or hpc/projects)
uv sync
```

You'll also need:
- CUDA + cuDNN if training with GPU acceleration.
- A [Weights & Biases](https://wandb.ai) account if you want online logging (optional — disabled by default, see `hpc/README.md`).
- ROS2 + a physical UR3e arm only if you're doing the real-robot deployment step in `jupyter/`.

## A note on secrets

None of the scripts in this repo should contain real API keys or credentials. Set `WANDB_API_KEY` (and anything similar) in your own shell profile (`~/.bashrc`) or an untracked `.env` file — never hardcode it in a script that gets committed.
