#!/bin/bash -l
#SBATCH --job-name=attention
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/attention_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/attention_%j.err
#SBATCH --time=02:00:00
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
    echo "sbatch analysis/attention.slurm <run_dir>"
    exit 1
fi

RUN_DIR=$1

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD
export CP_OUTPUT_ROOT=/scratch/creighton.jo/cellpainting

echo "RUN_DIR=$RUN_DIR"
echo "Starting attention map extraction job on $(hostname)"

python -u analysis/extract_attention_maps.py \
    --run_dir "$RUN_DIR" \
    "${@:2}"
