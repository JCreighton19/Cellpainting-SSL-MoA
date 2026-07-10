#!/bin/bash
#SBATCH --job-name=cp-download
#SBATCH --partition=sharing
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-23
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

# 24 (experiment, plate) pairs -- one per SLURM array task (--array=0-23
# above). This is the SAME diversity-maximizing selection
# scripts/download_compound_plates.py computes from
# analysis/outputs/compound_plates.csv (excluding the 9 acquisitions already
# downloaded -- see EXCLUDED_ACQUISITIONS there), captured here as a literal
# list so this legacy per-plate-array workflow stays usable without
# depending on the Python script at runtime. Every plate is still
# independently re-verified as a compound plate below regardless of what's
# written here -- this list is only which plates to ATTEMPT.
#
# All 24 are within cpg0000-jump-pilot, the only dataset currently
# configured in DATASET_CONFIGS (scripts/download_compound_plates.py) --
# NOT the 2020_11_18_CPJUMP1_TimepointDay1 / 2020_11_19_TimepointDay4 /
# 2020_12_02_CPJUMP1_2WeeksTimePoint experiments referenced by the old
# 8-plate version of this list. If you want plates from other datasets or
# experiments, regenerate with:
#   python3 -c "
#   import sys; sys.path.insert(0, 'scripts')
#   import download_compound_plates as dcp
#   for s in dcp.select_plates(dcp.load_candidate_plates(dcp.COMPOUND_PLATES_CSV), dcp.TARGET_NUM_PLATES):
#       print(f'  \"{s.experiment}:{s.barcode}\"')
#   "
#
# Kept in sync by hand with the identical list in download_metadata.slurm --
# both scripts are standalone (no shared bash library), so update both.
PLATES=(
  "2020_11_04_CPJUMP1:BR00117008"
  "2020_11_04_CPJUMP1_DL:BR00117010"
  "2020_12_08_CPJUMP1_Bleaching:BR00116991A"
  "2020_11_04_CPJUMP1:BR00117009"
  "2020_11_04_CPJUMP1_DL:BR00117011"
  "2020_12_08_CPJUMP1_Bleaching:BR00116991B"
  "2020_11_04_CPJUMP1:BR00117015"
  "2020_11_04_CPJUMP1_DL:BR00117012"
  "2020_12_08_CPJUMP1_Bleaching:BR00116991C"
  "2020_11_04_CPJUMP1:BR00117016"
  "2020_11_04_CPJUMP1_DL:BR00117013"
  "2020_12_08_CPJUMP1_Bleaching:BR00116991D"
  "2020_11_04_CPJUMP1:BR00117017"
  "2020_12_08_CPJUMP1_Bleaching:BR00116991E"
  "2020_11_04_CPJUMP1:BR00117019"
  "2020_12_08_CPJUMP1_Bleaching:BR00116991F"
  "2020_11_04_CPJUMP1:BR00117024"
  "2020_12_08_CPJUMP1_Bleaching:BR00116992A"
  "2020_11_04_CPJUMP1:BR00117025"
  "2020_12_08_CPJUMP1_Bleaching:BR00116992B"
  "2020_11_04_CPJUMP1:BR00117026"
  "2020_12_08_CPJUMP1_Bleaching:BR00116992C"
  "2020_11_04_CPJUMP1:BR00117050"
  "2020_12_08_CPJUMP1_Bleaching:BR00116992D"
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
