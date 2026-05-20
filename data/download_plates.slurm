#!/bin/bash
#SBATCH --job-name=cp-download
#SBATCH --partition=sharing
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/download_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/download_%j.err

set -euo pipefail

# Env setup and debugging info
echo "========================================"
echo "START TIME: $(date)"
echo "JOB ID: $SLURM_JOB_ID"
echo "NODE: $(hostname)"
echo "USER: $(whoami)"
echo "PWD: $(pwd)"
echo "========================================"

source ~/.bashrc
conda activate cellpainting

echo "Conda env: $CONDA_DEFAULT_ENV"
echo "AWS location: $(which aws || echo 'NOT FOUND')"
aws --version || true

# Config
BASE="s3://cellpainting-gallery/cpg0000-jump-pilot/source_4"
OUT="/scratch/creighton.jo/cellpainting/data/raw"

mkdir -p "$OUT"
mkdir -p /scratch/creighton.jo/cellpainting/logs

PLATES=(
BR00116991
BR00116992
BR00116993
BR00116994
BR00116995
)

# Helpers
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

fail_plate() {
    log "FAILED plate $1 (continuing)"
}

# Main loop
log "Starting Cell Painting download pipeline..."

for plate in "${PLATES[@]}"; do
    echo ""
    echo "========================================"
    log "Processing plate: $plate"
    echo "========================================"

    ### Resolve plate path safely
    log "Resolving S3 plate path..."

    PLATE_PATH=$(aws s3 ls \
        "${BASE}/images/2020_11_04_CPJUMP1/images/" \
        --no-sign-request \
        --only-show-errors \
        2>>/scratch/creighton.jo/cellpainting/logs/download_${SLURM_JOB_ID}.err \
        | awk -v p="$plate" '$2 ~ p {print $2}' | head -n 1 || true)

    if [[ -z "${PLATE_PATH}" ]]; then
        log "WARNING: Could not resolve plate path for $plate"
        fail_plate "$plate"
        continue
    fi

    log "Resolved plate path: $PLATE_PATH"

    # Images
    log "Syncing images..."

    if aws s3 sync \
        "${BASE}/images/2020_11_04_CPJUMP1/images/${PLATE_PATH}Images/" \
        "${OUT}/images/${plate}/" \
        --no-sign-request; then
        log "Images OK for $plate"
    else
        log "ERROR: Image sync failed for $plate"
        fail_plate "$plate"
        continue
    fi

    # Metadata (safe, shared)
    log "Syncing metadata (external + load data + platemaps)..."

    aws s3 sync \
        "${BASE}/workspace/metadata/external_metadata/" \
        "${OUT}/external_metadata/" \
        --no-sign-request || log "WARN: external_metadata sync issue"

    aws s3 sync \
        "${BASE}/workspace/load_data_csv/" \
        "${OUT}/load_data_csv/" \
        --no-sign-request || log "WARN: load_data_csv sync issue"

    aws s3 sync \
        "${BASE}/workspace/metadata/platemaps/2020_11_04_CPJUMP1/platemap/" \
        "${OUT}/platemaps/" \
        --no-sign-request || log "WARN: platemaps sync issue"

    log "Finished plate: $plate"

done

echo ""
log "ALL PLATES COMPLETE"
log "END TIME: $(date)"