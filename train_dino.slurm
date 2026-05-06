#!/bin/bash -l
#SBATCH --job-name=dino-cp
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p logs

cd ~/Cellpainting-SSL-MoA

echo "Using python:"
conda run -n cellpainting which python

echo "Testing torch:"
conda run -n cellpainting python -c "import torch; print(torch.__version__)"

conda run -n cellpainting python models/train_dino.py