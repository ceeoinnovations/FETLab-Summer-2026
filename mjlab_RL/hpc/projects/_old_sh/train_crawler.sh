#!/bin/bash
#SBATCH --job-name=lego_crawler
#SBATCH --output=crawler_out.log
#SBATCH --error=crawler_err.log
#SBATCH --time=01:00:00
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
from mjlab.scripts.train import main
sys.argv = ["train", "Mjlab-LegoCrawler-Forward",
            "--env.scene.num-envs", "128"]
main()
EOF