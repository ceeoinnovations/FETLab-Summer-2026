#!/bin/bash
#SBATCH --job-name=lego_quad_train17b_1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=49
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --exclude=pax166,pax017
#SBATCH --output=slurm_%j.out
#
# CPU/SB3 track, v7 reward iteration on the faithful MESH MODEL
# (lego_quad_mesh.xml, not the abstract crank model) - submit from this
# directory (cpu-projects/tasks/walk-straight/train17/):
#   sbatch submit_train17.sbatch
#
# 17 CPUs = 16 parallel envs (N_ENVS in train_deploy17.py) + 1 main process.
# The mesh model is ~2x slower than the abstract one (~556 fps at 8 envs);
# 16 envs should bring the 3M-step run to ~50-60 min; 2.5h for headroom.

module load uv/0.9.18
export UV_CACHE_DIR=/tmp/uv_cache_${SLURM_JOB_ID}
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1

set -e
cd "$SLURM_SUBMIT_DIR"   # train17/ ($0/dirname "$0" breaks under Slurm - it
                         # copies the script to a spool dir before running)
uv run python train_deploy17.py
