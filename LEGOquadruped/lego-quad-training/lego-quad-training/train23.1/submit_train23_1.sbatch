#!/bin/bash
#SBATCH --job-name=lego_quad_train23_1
#SBATCH --partition=batch
#SBATCH --cpus-per-task=49
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --time=05:00:00
#SBATCH --exclusive
#SBATCH --exclude=pax166,pax017
#SBATCH --output=slurm_%j.out
#
# CPU/SB3 track, v23.1: train23 (forward-gated diagonal coordination + back-active
# balance) + RECOVERY training (broken-phase RSI seeds, mid-episode phase kicks).
# Faithful MESH model. --exclusive: whole clean node (full ~1000+ fps).
# Submit from this directory:
#   sbatch submit_train23_1.sbatch
#
# 48 parallel envs (N_ENVS in train_deploy23_1.py) + main process.

module load uv/0.9.18
export UV_CACHE_DIR=/tmp/uv_cache_${SLURM_JOB_ID}
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1

set -e
cd "$SLURM_SUBMIT_DIR"   # train23.1/ ($0/dirname "$0" breaks under Slurm - it
                         # copies the script to a spool dir before running)
uv run python train_deploy23_1.py
