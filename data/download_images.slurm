#!/bin/bash
#SBATCH --job-name=cp-download
#SBATCH --partition=sharing
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-7
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/download_%A_%a.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/download_%A_%a.err

set -eo pipefail

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

echo "Python: $(which python)"
echo "AWS: $(which aws)"
aws --version

DATASET="cpg0000-jump-pilot"
BASE="s3://cellpainting-gallery/${DATASET}/source_4"
OUT="/scratch/creighton.jo/cellpainting/data/raw"
AWS_OPTS="--no-sign-request"

# 8 (experiment, plate) pairs, 2 per experiment across 4 experiments -- one
# per SLURM array task (--array=0-7 above). This list is only which plates
# to ATTEMPT; every plate is independently re-verified as a compound plate
# below regardless of what's written here.
#
# REPLACE_ME_* barcodes: only 2020_11_04_CPJUMP1's barcode_platemap.csv is
# already in this repo (data/raw/barcode_platemap.csv), so those two entries
# are real. The other three experiments' compound barcodes are not something
# to guess at -- fill them in from each experiment's own barcode_platemap.csv
# (same file this script downloads and checks against). A left-over
# REPLACE_ME_* will fail loudly in validation below, not silently proceed.
#
# Kept in sync by hand with the identical list in download_metadata.slurm --
# both scripts are standalone (no shared bash library), so update both.
PLATES=(
  "2020_11_04_CPJUMP1:BR00116991"
  "2020_11_04_CPJUMP1:BR00116992"
  "2020_11_18_CPJUMP1_TimepointDay1:REPLACE_ME_1"
  "2020_11_18_CPJUMP1_TimepointDay1:REPLACE_ME_2"
  "2020_11_19_TimepointDay4:REPLACE_ME_3"
  "2020_11_19_TimepointDay4:REPLACE_ME_4"
  "2020_12_02_CPJUMP1_2WeeksTimePoint:REPLACE_ME_5"
  "2020_12_02_CPJUMP1_2WeeksTimePoint:REPLACE_ME_6"
)

pair="${PLATES[$SLURM_ARRAY_TASK_ID]}"
experiment="${pair%%:*}"
plate="${pair##*:}"

echo "========================================"
echo "Task: ${SLURM_ARRAY_TASK_ID}"
echo "Experiment: ${experiment}"
echo "Plate: ${plate}"
echo "Start: $(date)"
echo "========================================"

# --- Compound-plate validation -------------------------------------------
# Sets global PLATE_MAP_NAME on success, returns nonzero on any failure.
# Must be called directly (not inside command substitution) -- `return`
# here needs to propagate to the caller's own `exit`, which a subshell
# created by $(...) would swallow.
#
# Identical to the function of the same name in download_metadata.slurm --
# both scripts are standalone, so keep them in sync by hand.
resolve_plate_map_name() {
    local experiment="$1"
    local plate="$2"
    local barcode_csv_dir="${OUT}/platemaps/${DATASET}/${experiment}"
    local barcode_csv="${barcode_csv_dir}/barcode_platemap.csv"

    mkdir -p "$barcode_csv_dir"
    if [[ ! -s "$barcode_csv" ]]; then
        if ! aws s3 cp \
              "${BASE}/workspace/metadata/platemaps/${experiment}/barcode_platemap.csv" \
              "$barcode_csv" \
              $AWS_OPTS; then
            echo "ERROR: failed to download barcode_platemap.csv for experiment ${experiment}" >&2
            rm -f "$barcode_csv"
            return 1
        fi
    fi

    PLATE_MAP_NAME=$(tr -d '\r' < "$barcode_csv" \
        | awk -F',' -v p="$plate" '$1 == p {print $2; found=1} END{exit !found}')

    if [[ -z "$PLATE_MAP_NAME" ]]; then
        echo "ERROR: plate ${plate} not found in ${experiment}'s barcode_platemap.csv" >&2
        return 1
    fi
    return 0
}

if ! resolve_plate_map_name "$experiment" "$plate"; then
    echo "ABORT: could not resolve plate type for ${experiment}/${plate}."
    exit 1
fi

echo "Plate_Map_Name: ${PLATE_MAP_NAME}"

if [[ "$PLATE_MAP_NAME" != "JUMP-Target-1_compound_platemap" ]]; then
    echo "ERROR: ${experiment}/${plate} is NOT a compound plate (Plate_Map_Name=${PLATE_MAP_NAME})."
    echo "ABORT: refusing to download a non-compound (ORF/CRISPR/other) plate."
    exit 1
fi

echo "Verified compound plate: ${experiment}/${plate}"

# NOTE: this script does not write data/raw/experiment_datasets.json (the
# experiment -> dataset manifest datasets/build_metadata_table.py requires).
# It's written by download_metadata.slurm instead, to avoid multiple
# parallel array tasks here doing a concurrent, non-atomic read-modify-write
# on the same JSON file. Run download_metadata.slurm at least once for every
# experiment downloaded here.

# --- Resolve the full acquisition folder name -----------------------------
# The S3 folder name is barcode + timestamp + measurement number, not the
# bare barcode (the same barcode can be re-imaged under a different
# acquisition at another timepoint/experiment) -- anchored on "^{plate}__"
# rather than a loose substring match so one barcode can't accidentally
# match another that merely contains it.
PLATE_PATH=$(aws s3 ls \
  "${BASE}/images/${experiment}/images/" \
  $AWS_OPTS \
  | awk -v p="$plate" '$2 ~ "^"p"__" {print $2}')

if [[ -z "$PLATE_PATH" ]]; then
    echo "ERROR: could not resolve acquisition folder for ${experiment}/${plate}"
    exit 1
fi

if [[ $(echo "$PLATE_PATH" | wc -l) -gt 1 ]]; then
    echo "ERROR: ambiguous acquisition folders for ${experiment}/${plate}:"
    echo "$PLATE_PATH"
    exit 1
fi

acquisition_id="${PLATE_PATH%/}"
echo "Resolved acquisition: ${acquisition_id}"

# aws s3 sync only transfers missing/changed files, so an interrupted
# download resumes correctly on rerun without re-transferring what's already
# there and without needing any extra bookkeeping here.
aws s3 sync \
  "${BASE}/images/${experiment}/images/${acquisition_id}/Images/" \
  "${OUT}/images/${experiment}/${acquisition_id}/" \
  $AWS_OPTS

echo "Done: ${experiment}/${plate}"
echo "End: $(date)"
