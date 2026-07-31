#!/bin/bash
#SBATCH --job-name=ur3e_reach
#SBATCH --output=ur3e_reach_out.log
#SBATCH --error=ur3e_reach_err.log
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

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
sys.argv = ["play", "Mjlab-UR3e-Reach",
            "--checkpoint-file", "logs/rsl_rl/ur3e_reach/2026-07-09_17-01-40/model_1999.pt"]
main()
EOF