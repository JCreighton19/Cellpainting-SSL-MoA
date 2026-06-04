#!/bin/bash -l
#SBATCH --job-name=prep-cp
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
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

OUT_DIR=/scratch/creighton.jo/cellpainting/processed/tiles

echo "Clearing old tiles in: $OUT_DIR"
rm -rf $OUT_DIR
mkdir -p $OUT_DIR

echo "Starting preprocessing..."

python -u datasets/preprocess_dataset.py