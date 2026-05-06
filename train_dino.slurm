#!/bin/bash -l
#SBATCH --job-name=dino-cp
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e  # fail fast if anything breaks

mkdir -p logs
cd ~/Cellpainting-SSL-MoA

echo "Initializing conda..."

# Load conda properly (more reliable than eval-only usage)
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh

echo "Activating env..."
conda activate cellpainting

echo "Python used:"
which python
python --version

echo "Conda env:"
echo $CONDA_PREFIX

echo "Torch check:"
python -c "import torch; print(torch.__version__)"

echo "Starting training..."
python models/train_dino.py