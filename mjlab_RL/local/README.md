# local/

Local (non-cluster) version of the mjlab project, for fast iteration before submitting a training run on `hpc/`: small `num_envs`, MuJoCo's interactive viewer, and quick test scripts against a fixed target instead of a full training loop.

## Layout

```
local/
└── my_mjlab_project/
    ├── tasks/
    │   ├── ur3e_reach/      # UR3e arm reach task
    │   │   ├── test_ur3e.py          # standalone MuJoCo viewer script, no RL — sanity-checks the arm model/IK against a hardcoded joint trajectory
    │   │   └── ur3e_sim_out_test.py  # runs a trained policy in closed loop toward one fixed target, logs joint pos/action for comparison against a real-robot run
    │   ├── silly_walk/       # LEGO crawler task (mirrors hpc/projects/my_tasks/silly_walk)
    │   └── template/         # skeleton for scaffolding a new task — see below
    └── __init__.py
```

The task code in `tasks/ur3e_reach/` and `tasks/silly_walk/` is the same as in `hpc/projects/my_tasks/` — this is the local copy used for development before a task is run on the cluster. Keep both in sync when you change one, or promote this folder to be the single source of truth and have `hpc/` reference it (not currently set up that way).

## Using `template/`

`tasks/template/` is **not runnable as-is** — it's a checklist/skeleton for creating a new task, distilled from the two working examples. To make a new task:

1. Copy `template/` to `tasks/<your_task_name>/`.
2. Fill in every `TODO` in `example_env_cfg.py`, `commands.py`, and `__init__.py` (task ID, experiment name, hidden layer sizes, etc).
3. Register a **unique** `task_id` string in `__init__.py` — two tasks sharing the same ID will silently break the import of the *entire* package, not just that task (see the comment at the bottom of `template/__init__.py` for what that failure looks like).
4. Add an import for your new task in the project's top-level `__init__.py`.

Before trusting any field name or class from `mjlab` referenced in the template, verify it against your installed mjlab version (the template has notes on how — e.g. grepping `mjlab/entity/data.py` for available field names).

## Requirements

- `mjlab` and its dependencies (see main [README](../README.md) for `uv` setup).
- MuJoCo's interactive viewer (`mujoco.viewer`) for `test_ur3e.py`.
