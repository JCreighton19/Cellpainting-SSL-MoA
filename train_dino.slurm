#!/bin/bash -l
#SBATCH --job-name=dino-cp
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -e

mkdir -p logs
cd ~/Cellpainting-SSL-MoA

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

# use explicit path — bypasses any shell resolution issues
PYTHON=/home/creighton.jo/.conda/envs/cellpainting/bin/python

echo "Python used:"
which $PYTHON
$PYTHON --version

echo "Torch check:"
$PYTHON -c "import torch; print(torch.__version__)"

echo "Starting training..."
$PYTHON models/train_dino.py