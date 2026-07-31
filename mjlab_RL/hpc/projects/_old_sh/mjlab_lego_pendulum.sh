#!/bin/bash
#SBATCH --job-name=lego_pendulum
#SBATCH --output=lego_pendulum_out.log
#SBATCH --error=lego_pendulum_err.log
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

export UV_CACHE_DIR=/tmp/uv_cache_${SLURM_JOB_ID}
mkdir -p $UV_CACHE_DIR
export UV_LINK_MODE=copy
export WANDB_API_KEY=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
export PYTHONUNBUFFERED=1

module load cuda
module load cudnn
module load python
module load uv

cd /cluster/home/lhanne01/mjlab_projects

uv run python -u - <<'EOF'
import sys
sys.path.insert(0, "/cluster/home/lhanne01/mjlab_projects/projects")
import mjlab
import my_tasks
from mjlab.scripts.play import main
sys.argv = ["play", "Mjlab-LegoPendulum-Swingup",
            "--checkpoint-file", "logs/rsl_rl/lego_pendulum/2026-06-26_14-46-24/model_499.pt"]
main()
EOF