#!/bin/bash
#SBATCH --job-name=cp-download
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-4
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/download_%A_%a.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/download_%A_%a.err

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

echo "Python: $(which python)"
echo "AWS: $(which aws)"
aws --version

BASE="s3://cellpainting-gallery/cpg0000-jump-pilot/source_4"
OUT="/scratch/creighton.jo/cellpainting/data/raw"

AWS="--no-sign-request --only-show-errors"

PLATES=(
BR00116991
BR00116992
BR00116993
BR00116994
BR00116995
)

plate=${PLATES[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Plate: $plate"
echo "Start: $(date)"
echo "========================================"

PLATE_PATH=$(aws s3 ls \
  "${BASE}/images/2020_11_04_CPJUMP1/images/" \
  $AWS \
  | awk -v p="$plate" '$2 ~ p {print $2}' | head -n 1)

if [[ -z "$PLATE_PATH" ]]; then
    echo "ERROR: could not resolve path for $plate"
    exit 1
fi

echo "Resolved: $PLATE_PATH"

aws s3 sync \
  "${BASE}/images/2020_11_04_CPJUMP1/images/${PLATE_PATH}Images/" \
  "${OUT}/images/${plate}/" \
  $AWS

echo "Done $plate"
echo "End: $(date)"