#!/bin/bash -l
#SBATCH --job-name=embed
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/embed_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/embed_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

set -euo pipefail

mkdir -p /scratch/creighton.jo/cellpainting/logs

# ---- REQUIRED ARG ----
if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "sbatch analysis/embed.slurm <run_dir>"
    exit 1
fi

RUN_DIR=$1

DATA_ROOT=/scratch/creighton.jo/cellpainting/data/raw
OUT_ROOT=/scratch/creighton.jo/cellpainting

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD
export CP_OUTPUT_ROOT=/scratch/creighton.jo/cellpainting

echo "RUN_DIR=$RUN_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "Starting embedding job on $(hostname)"

python -u analysis/extract_embeddings.py \
    --run_dir "$RUN_DIR" \
    "${@:2}"