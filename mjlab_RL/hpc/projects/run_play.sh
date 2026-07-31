#!/bin/bash
# Call via ./submit_play.sh, not `sbatch run_play.sh ...` directly.
#
# Positional args:
#   $1 = task_name  (e.g. Mjlab-UR3e-Reach)
#   $2 = job_name   (the training run's job name -- used to find
#                    logs/rsl_rl/<job_name>/<latest timestamp>/model_<latest>.pt)

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
# job-name / time / output / error deliberately not set here -- submit_play.sh
# passes them as sbatch flags.

set -euo pipefail

TASK_NAME="${1:?Usage: run_play.sh <task_name> <job_name>}"
JOB_NAME="${2:?Missing job_name}"

cd /cluster/home/lhanne01/mjlab_projects

CHECKPOINT_ROOT="logs/rsl_rl/${JOB_NAME}"

# Latest timestamped run directory. Directory names are ISO-8601-like
# (YYYY-MM-DD_HH-MM-SS), so plain lexicographic sort gives correct
# chronological order.
LATEST_DIR=$(ls -1d "${CHECKPOINT_ROOT}"/*/ 2>/dev/null | sort | tail -n 1 || true)
if [ -z "${LATEST_DIR}" ]; then
  echo "No run directories found under ${CHECKPOINT_ROOT}" >&2
  exit 1
fi

# Highest-iteration checkpoint in that directory. Using sort -V (version/
# natural sort) instead of plain sort, since plain lexicographic sort would
# incorrectly place model_500.pt after model_1999.pt (string comparison,
# '5' > '1').
LATEST_MODEL=$(ls -1 "${LATEST_DIR}"model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
if [ -z "${LATEST_MODEL}" ]; then
  echo "No model_*.pt checkpoints found in ${LATEST_DIR}" >&2
  exit 1
fi

echo "Task: ${TASK_NAME}"
echo "Using checkpoint: ${LATEST_MODEL}"

export UV_CACHE_DIR=/tmp/uv_cache_${SLURM_JOB_ID}
mkdir -p "$UV_CACHE_DIR"
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1

module load cuda
module load cudnn
module load python
module load uv

uv run python -u - "$TASK_NAME" "$LATEST_MODEL" <<'EOF'
import sys
sys.path.insert(0, "/cluster/home/lhanne01/mjlab_projects/projects")
import mjlab
import my_tasks
from mjlab.scripts.play import main

task_name, checkpoint_file = sys.argv[1], sys.argv[2]
sys.argv = ["play", task_name, "--checkpoint-file", checkpoint_file]
main()
EOF