#!/bin/bash
#SBATCH --job-name=cp-download
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

source ~/.bashrc
conda activate cellpainting

set -euo pipefail

echo "Running on: $(hostname)"
echo "Start time: $(date)"

echo "Starting full Cell Painting plate download..."

BASE="s3://cellpainting-gallery/cpg0000-jump-pilot/source_4"

OUT="$HOME/Cellpainting-SSL-MoA/data/raw"
mkdir -p "$OUT"

PLATES=(
BR00116991
BR00116992
BR00116993
BR00116994
BR00116995
)

for plate in "${PLATES[@]}"; do
    echo "======================================"
    echo "Downloading plate: $plate"

    # Resolve full S3 directory name
    PLATE_PATH=$(aws s3 ls \
    "${BASE}/images/2020_11_04_CPJUMP1/images/" \
    --no-sign-request | grep "${plate}" | awk '{print $2}')

    # 1. IMAGES
    echo "Images..."
    aws s3 sync \
        "${BASE}/images/2020_11_04_CPJUMP1/images/${PLATE_PATH}Images/" \
        "${OUT}/images/${plate}/" \
        --no-sign-request

    # 2. COMPOUND METADATA
    echo "Compound metadata..."
    aws s3 sync \
      "${BASE}/workspace/metadata/external_metadata/" \
      "${OUT}/external_metadata/" \
      --no-sign-request

    # 3. LOAD DATA CSV (correct location)
    echo "Load data..."
    aws s3 sync \
        "${BASE}/workspace/load_data_csv/" \
        "${OUT}/load_data_csv/" \
        --no-sign-request

    # 4. HARMONIZED METADATA (optional but useful)
    echo "Platemaps..."
    aws s3 sync \
    "${BASE}/workspace/metadata/platemaps/2020_11_04_CPJUMP1/platemap/" \
    "${OUT}/platemaps/" \
    --no-sign-request

    echo "Finished $plate"
done

echo "ALL PLATES COMPLETE"