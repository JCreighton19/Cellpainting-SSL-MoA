#!/bin/bash
#SBATCH --job-name=cp-download
#SBATCH --partition=sharing
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

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

    # 1. IMAGES
    echo "Images..."
    aws s3 sync \
        "${BASE}/images/2020_11_04_CPJUMP1/images/${plate}__*/Images/" \
        "${OUT}/images/${plate}/" \
        --no-sign-request

    # 2. COMPOUND METADATA
    echo "Compound metadata..."
    aws s3 sync \
        "${BASE}/compound_metadata/${plate}/" \
        "${OUT}/compound_metadata/${plate}/" \
        --no-sign-request

    # 3. LOAD DATA CSV
    echo "Load data..."
    aws s3 sync \
        "${BASE}/load_data_csv/${plate}/" \
        "${OUT}/load_data_csv/${plate}/" \
        --no-sign-request

    # 4. PLATEMAP
    echo "Platemap..."
    aws s3 sync \
        "${BASE}/platemaps/${plate}/" \
        "${OUT}/platemaps/${plate}/" \
        --no-sign-request

    echo "Finished $plate"
done

echo "ALL PLATES COMPLETE"