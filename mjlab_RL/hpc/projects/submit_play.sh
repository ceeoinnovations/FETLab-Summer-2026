#!/bin/bash
# Usage:
#   ./submit_play.sh <job_name> <task_name>
#
# Example:
#   ./submit_play.sh ur3e_reach Mjlab-UR3e-Reach
#
# job_name should match the ORIGINAL training run's job name -- that's
# what run_play.sh uses to find logs/rsl_rl/<job_name>/.../model_*.pt.
#
# Time is fixed at 20 minutes -- play just runs inference on a trained
# checkpoint, no reason for this to vary run to run.
#
# Logs always go to training_logs/play_out.log and training_logs/play_err.log
# -- fixed names, overwritten each run

set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <job_name> <task_name>"
  exit 1
fi

JOB_NAME=$1
TASK_NAME=$2
TIME="00:20:00"

mkdir -p training_logs
OUTPUT="training_logs/play_out.log"
ERROR="training_logs/play_err.log"

sbatch \
  --job-name="${JOB_NAME}_play" \
  --time="${TIME}" \
  --output="${OUTPUT}" \
  --error="${ERROR}" \
  run_play.sh "${TASK_NAME}" "${JOB_NAME}"