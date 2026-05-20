#!/bin/bash -l
#SBATCH --job-name=umap
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/umap_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/umap_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

set -euo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

# Paths
ROOT=/scratch/creighton.jo/cellpainting
EMB=$ROOT/embeddings
OUT=$ROOT/figures
LOGS=$ROOT/logs

mkdir -p "$OUT"
mkdir -p "$LOGS"

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD

echo "ROOT=$ROOT"
echo "EMBEDDINGS=$EMB"
echo "OUTPUT=$OUT"

python analysis/run_umap.py \
    --input "$EMB" \
    --output "$OUT"