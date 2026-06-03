#!/bin/bash -l
#SBATCH --job-name=scvg-cp
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/scvg_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/scvg_%j.err
#SBATCH --time=05:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -e

# Paths
REPO=/home/creighton.jo/Cellpainting-SSL-MoA
DATA=/scratch/creighton.jo/cellpainting/data/raw
OUT=/scratch/creighton.jo/cellpainting

mkdir -p /scratch/creighton.jo/cellpainting/logs

cd $REPO

# Python path
export PYTHONPATH=$REPO

# Tell dataset where data lives
export CP_DATA_ROOT=$DATA
export CP_OUTPUT_ROOT=$OUT
export HF_HOME=/scratch/creighton.jo/hf_cache
export TORCH_HOME=/scratch/creighton.jo/torch_cache

# Conda setup
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

PYTHON=$(which python)

echo "Python used:"
$PYTHON -c "import sys; print(sys.executable)"
$PYTHON --version

echo "Torch check:"
$PYTHON -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

echo "Data root:"
echo $CP_DATA_ROOT

echo "Starting training..."

$PYTHON models/scvg/train_scvg.py