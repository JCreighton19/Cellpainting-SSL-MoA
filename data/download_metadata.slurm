#!/bin/bash
#SBATCH --job-name=cp-metadata
#SBATCH --partition=sharing
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/meta_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/meta_%j.err

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
unset MKL_INTERFACE_LAYER || true
conda activate cellpainting

echo "Python: $(which python)"
echo "AWS: $(which aws)"
aws --version

BASE="s3://cellpainting-gallery/cpg0000-jump-pilot/source_4"
OUT="/scratch/creighton.jo/cellpainting/data/raw"
AWS_OPTS="--no-sign-request --only-show-errors"

mkdir -p "$OUT"

echo "Downloading GLOBAL metadata (once)..."

# 1. GLOBAL compound metadata-
aws s3 cp \
  "${BASE}/workspace/metadata/external_metadata/JUMP-Target-1_compound_metadata.tsv" \
  "${OUT}/JUMP-Target-1_compound_metadata.tsv" \
  $AWS_OPTS || true

aws s3 cp \
  "${BASE}/workspace/metadata/external_metadata/JUMP-Target-1_compound_metadata_targets.tsv" \
  "${OUT}/JUMP-Target-1_compound_metadata_targets.tsv" \
  $AWS_OPTS || true

# 2. GLOBAL compound platemap
mkdir -p "${OUT}/platemaps/2020_11_04_CPJUMP1"

aws s3 cp \
  "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/metadata/platemaps/2020_11_04_CPJUMP1/platemap/JUMP-Target-1_compound_platemap.txt" \
  "${OUT}/platemaps/2020_11_04_CPJUMP1/JUMP-Target-1_compound_platemap.txt" \
  $AWS_OPTS || true

# 3. PLATE load_data_csv (experiment-scoped)
PLATES=(
  BR00116991
  BR00116992
  BR00116993
  BR00116994
  BR00116995
)

EXPERIMENT="2020_11_04_CPJUMP1"

echo "Downloading plate-level load_data_csv..."

for PLATE in "${PLATES[@]}"; do
    echo "== $PLATE =="

    mkdir -p "${OUT}/load_data_csv/${PLATE}"

    aws s3 cp \
      "${BASE}/workspace/load_data_csv/${EXPERIMENT}/${PLATE}/load_data.csv" \
      "${OUT}/load_data_csv/${PLATE}/load_data.csv" \
      $AWS_OPTS || true

done

echo "Metadata complete"