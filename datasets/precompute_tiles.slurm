#!/bin/bash -l
#SBATCH --job-name=tiles-cp
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/tiles_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/tiles_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

REPO=/home/creighton.jo/Cellpainting-SSL-MoA

mkdir -p /scratch/creighton.jo/cellpainting/logs

cd $REPO

export PYTHONPATH=$REPO
export CP_OUTPUT_ROOT=/scratch/creighton.jo/cellpainting

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

echo "Python used:"
which python
python --version

echo "Starting tile precomputation..."

python -u datasets/precompute_tiles.py \
    --n_tiles 20 \
    --crop_size 224 \
    --max_iou 0.5 \
    --fp16

echo "Tile precomputation complete."
