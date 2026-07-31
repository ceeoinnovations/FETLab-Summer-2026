#!/bin/bash
#SBATCH --job-name=mjlab_test
#SBATCH --output=mjlab_out.log
#SBATCH --error=mjlab_err.log
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

export UV_CACHE_DIR=/tmp/uv_cache_${SLURM_JOB_ID}
mkdir -p $UV_CACHE_DIR

module load cuda
module load cudnn

module load python
module load uv
uv run demo
