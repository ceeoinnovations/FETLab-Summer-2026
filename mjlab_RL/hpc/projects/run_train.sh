#!/bin/bash
# The actual SLURM job. Call via ./submit_train.sh, not `sbatch run_train.sh ...`
# directly -- submit.sh sets job-name/time/output/error as real sbatch
# flags (see submit.sh's comment for why that has to happen there, not
# here as #SBATCH lines).
#
# Positional args:
#   $1 = task_name
#   $2 = num_envs
#   $3 = iterations (may be an empty string -- means "don't set it at all")

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
# job-name / time / output / error deliberately not set here -- submit.sh
# passes them as sbatch flags, which take precedence over this file anyway.

set -euo pipefail

TASK_NAME="${1:?Usage: run_train.sh <task_name> <num_envs> [iterations]}"
NUM_ENVS="${2:?Missing num_envs}"
ITERATIONS="${3:-}"   # empty string means "no iterations override"

echo "Task: ${TASK_NAME}, num_envs: ${NUM_ENVS}, iterations: ${ITERATIONS:-<not set>}"

export UV_CACHE_DIR=/tmp/uv_cache_${SLURM_JOB_ID}
mkdir -p "$UV_CACHE_DIR"
export UV_LINK_MODE=copy
# Set WANDB_API_KEY in your shell profile (~/.bashrc) or a separate
# untracked file instead of hardcoding it here.

# if using wandb, uncomment the following line and set your API key:
# export WANDB_API_KEY=your_wandb_api_key_here
export WANDB_MODE=disabled  # comment out to enable online logging
export PYTHONUNBUFFERED=1

module load cuda
module load cudnn
module load python
module load uv

cd /cluster/home/lhanne01/mjlab_projects

uv run python -u - "$TASK_NAME" "$NUM_ENVS" "$ITERATIONS" <<'EOF'
import sys
sys.path.insert(0, "/cluster/home/lhanne01/mjlab_projects/projects")
import mjlab
import my_tasks
from mjlab.scripts.train import main

task_name, num_envs, iterations = sys.argv[1], sys.argv[2], sys.argv[3]

sys.argv = ["train", task_name, "--env.scene.num-envs", num_envs, "--gpu-ids", "all"]
if iterations:  # empty string -> skip the flag entirely
    sys.argv += ["--agent.max-iterations", iterations]

main()
EOF