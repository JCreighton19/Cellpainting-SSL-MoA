#!/bin/bash -l
#SBATCH --job-name=prep-cp
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/prep_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/prep_%j.err

set -e

mkdir -p /scratch/creighton.jo/cellpainting/logs

# conda (IMPORTANT: same as training script)
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

export CP_OUTPUT_ROOT=/scratch/creighton.jo/cellpainting
export CP_DATA_ROOT=/scratch/creighton.jo/cellpainting/data/raw

echo "Python:"
which python
python --version

echo "Starting preprocessing..."

python -u datasets/preprocess_dataset.py