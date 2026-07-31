#!/bin/bash
# Usage:
#   ./submit_train.sh <job_name> <time> <task_name> <num_envs> [iterations]
#
# Examples:
#   ./submit_train.sh ur3e_reach 01:00:00 Mjlab-UR3e-Reach 2048 2000
#   ./submit_train.sh ur3e_reach 01:00:00 Mjlab-UR3e-Reach 2048        # no iterations flag at all
#
# output/error logs are auto-derived as training_logs/<job_name>_out.log /
# training_logs/<job_name>_err.log -- a new pair of logs each run, since
# job_name typically changes (or the folder just accumulates one pair per
# distinct job_name used).
#
# job-name/output/error/time become #SBATCH directives, which SLURM reads
# from the file at submission time (before any shell runs), so they can't
# come from run_train.sh's own positional args -- they're passed as real sbatch
# flags here instead. task_name/num_envs/iterations are just script-body
# logic (building a Python command), so those work fine as positional args.

set -euo pipefail

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 <job_name> <time> <task_name> <num_envs> [iterations]"
  exit 1
fi

JOB_NAME=$1
TIME=$2
TASK_NAME=$3
NUM_ENVS=$4
ITERATIONS=${5:-}   # empty string if not provided

mkdir -p training_logs
OUTPUT="training_logs/${JOB_NAME}_out.log"
ERROR="training_logs/${JOB_NAME}_err.log"

sbatch \
  --job-name="${JOB_NAME}" \
  --time="${TIME}" \
  --output="${OUTPUT}" \
  --error="${ERROR}" \
  run_train.sh "${TASK_NAME}" "${NUM_ENVS}" "${ITERATIONS}"