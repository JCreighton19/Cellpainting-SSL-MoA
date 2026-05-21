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
conda activate cellpainting

echo "Python: $(which python)"
echo "AWS: $(which aws)"
aws --version

BASE="s3://cellpainting-gallery/cpg0000-jump-pilot/source_4"
OUT="/scratch/creighton.jo/cellpainting/data/raw"

mkdir -p "$OUT"

AWS_OPTS="--no-sign-request --only-show-errors"

echo "Downloading metadata..."

aws s3 sync \
  "${BASE}/workspace/metadata/external_metadata/" \
  "${OUT}/external_metadata/" \
  $AWS

aws s3 sync \
  "${BASE}/workspace/load_data_csv/" \
  "${OUT}/load_data_csv/" \
  $AWS

aws s3 sync \
  "${BASE}/workspace/metadata/platemaps/2020_11_04_CPJUMP1/platemap/" \
  "${OUT}/platemaps/" \
  $AWS

echo "Metadata complete"