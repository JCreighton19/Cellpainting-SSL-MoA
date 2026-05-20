#!/bin/bash -l
#SBATCH --job-name=embed
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/embed_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/embed_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

mkdir -p /scratch/creighton.jo/cellpainting/logs

DATA_ROOT=/scratch/creighton.jo/cellpainting/data/raw
OUT_ROOT=/scratch/creighton.jo/cellpainting

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD

echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "Starting embedding job on $(hostname)"

python analysis/extract_embeddings.py \
    --data_root "$DATA_ROOT" \
    --out_dir "$OUT_ROOT/embeddings"