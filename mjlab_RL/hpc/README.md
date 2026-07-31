# hpc/

SLURM cluster scripts and task code for training/running mjlab RL tasks on a GPU partition.

> **This folder is written for our specific lab cluster.** If you're adapting it for another cluster, see "What to change" below.

## Layout

```
hpc/
├── projects/
│   ├── my_tasks/            # task definitions (env cfgs, commands, PPO configs)
│   │   ├── ur3e_reach/
│   │   └── silly_walk/
│   ├── submit_train.sh      # entry point: sets #SBATCH flags, calls run_train.sh
│   ├── run_train.sh          # the actual SLURM job body for training
│   ├── submit_play.sh       # entry point for running/evaluating a checkpoint
│   ├── run_play.sh           # the actual SLURM job body for play/eval
│   └── _old_sh/              # earlier, one-off SLURM scripts, kept for reference
```

## Usage

Train:
```bash
cd hpc/projects
./submit_train.sh <job_name> <time> <task_name> <num_envs> [iterations]

# example
./submit_train.sh ur3e_reach 01:00:00 Mjlab-UR3e-Reach 2048 2000
```

Play back / evaluate the latest checkpoint from a training run:
```bash
./submit_play.sh <job_name> <task_name>

# example
./submit_play.sh ur3e_reach Mjlab-UR3e-Reach
```

`submit_train.sh`/`submit_play.sh` set `--job-name`, `--time`, `--output`, `--error` as real `sbatch` flags (SLURM needs these at submission time, before any shell in `run_*.sh` executes) and then call `run_train.sh`/`run_play.sh`, which do the actual work. Always call the `submit_*.sh` scripts — don't `sbatch run_train.sh` directly.

Logs land in `training_logs/<job_name>_out.log` / `_err.log`. Checkpoints land in `logs/rsl_rl/<job_name>/<timestamp>/model_<iteration>.pt`; `run_play.sh` automatically finds the latest run and highest-iteration checkpoint for a given `job_name`.

### Logging to Weights & Biases

Online W&B logging is **off by default** (`WANDB_MODE=disabled` in `run_train.sh`). To enable it, set your own key in `~/.bashrc` (or an untracked file) — never hardcode it in a script:
```bash
export WANDB_API_KEY=your_key_here
```
Then comment out the `WANDB_MODE=disabled` line in `run_train.sh`.

## What to change for a different cluster

- `run_train.sh` / `run_play.sh`: the `cd /cluster/home/lhanne01/mjlab_projects` line and the `module load ...` lines (`cuda`, `cudnn`, `python`, `uv`) are specific to our cluster's module system and our home directory — swap in your own path and whatever module names your cluster uses.
- The `#SBATCH --partition=gpu` line assumes a partition named `gpu` exists; check `sinfo` on your cluster.

## `_old_sh/`

Earlier one-off SLURM scripts (`train_ur3e_reach.sh`, `train_crawler.sh`, `play_ur3e_reach.sh`, `mjlab_demo.sh`, `mjlab_lego_pendulum.sh`), kept for reference. Superseded by the `submit_*.sh` / `run_*.sh` split above, which separates SLURM directives that must be known at submit time from the actual job logic.
