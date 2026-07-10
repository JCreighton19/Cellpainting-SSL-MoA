#!/bin/bash
#SBATCH --job-name=cp-metadata
#SBATCH --partition=sharing
#SBATCH --time=03:00:00
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

DATASET="cpg0000-jump-pilot"
BASE="s3://cellpainting-gallery/${DATASET}/source_4"
OUT="/scratch/creighton.jo/cellpainting/data/raw"
AWS_OPTS="--no-sign-request --only-show-errors"

mkdir -p "$OUT"

# Same 24 (experiment, plate) pairs as download_images.slurm -- kept in sync
# by hand (see comment there for provenance/regeneration command). Each is
# re-verified as a compound plate below before its load_data.csv is
# downloaded; ORF/CRISPR plates are never fetched even if one somehow ends
# up in this list.
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

echo "Downloading GLOBAL compound metadata (once)..."

aws s3 cp \
  "${BASE}/workspace/metadata/external_metadata/JUMP-Target-1_compound_metadata.tsv" \
  "${OUT}/JUMP-Target-1_compound_metadata.tsv" \
  $AWS_OPTS

aws s3 cp \
  "${BASE}/workspace/metadata/external_metadata/JUMP-Target-1_compound_metadata_targets.tsv" \
  "${OUT}/JUMP-Target-1_compound_metadata_targets.tsv" \
  $AWS_OPTS

# --- Compound-plate validation -------------------------------------------
# Identical to the function of the same name in download_images.slurm --
# both scripts are standalone, so keep them in sync by hand. See that file
# for why `return` (not `exit`) and direct invocation (not `$(...)`) matter.
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

# datasets/build_metadata_table.py needs to know which dataset each
# experiment came from (images/load_data_csv paths don't carry dataset at
# all) -- this is the only place that mapping is recorded. Safe to call once
# per plate here since this script is a single serial job, not a parallel
# array like download_images.slurm (which deliberately does NOT write this
# file -- see the note there).
update_experiment_dataset_manifest() {
    local experiment="$1"
    local dataset="$2"
    local manifest="${OUT}/experiment_datasets.json"
    if ! python - "$manifest" "$experiment" "$dataset" <<'PYEOF'
import json, sys
from pathlib import Path

path, experiment, dataset = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = json.loads(path.read_text()) if path.exists() else {}
existing = manifest.get(experiment)
if existing not in (None, dataset):
    sys.exit(f"experiment {experiment!r} already mapped to dataset {existing!r}, cannot also map to {dataset!r}")
if existing != dataset:
    manifest[experiment] = dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
PYEOF
    then
        echo "ERROR: failed to update experiment_datasets.json for ${experiment}" >&2
        return 1
    fi
    return 0
}

for pair in "${PLATES[@]}"; do
    experiment="${pair%%:*}"
    plate="${pair##*:}"

    echo "========================================"
    echo "Experiment: ${experiment}"
    echo "Plate: ${plate}"
    echo "Start: $(date)"
    echo "========================================"

    if ! resolve_plate_map_name "$experiment" "$plate"; then
        echo "ABORT: could not resolve plate type for ${experiment}/${plate}."
        exit 1
    fi

    echo "Plate_Map_Name: ${PLATE_MAP_NAME}"

    if [[ "$PLATE_MAP_NAME" != "JUMP-Target-1_compound_platemap" ]]; then
        echo "ERROR: ${experiment}/${plate} is NOT a compound plate (Plate_Map_Name=${PLATE_MAP_NAME})."
        echo "ABORT: refusing to download metadata for a non-compound (ORF/CRISPR/other) plate."
        exit 1
    fi

    echo "Verified compound plate: ${experiment}/${plate}"

    if ! update_experiment_dataset_manifest "$experiment" "$DATASET"; then
        exit 1
    fi

    # Experiment-specific compound platemap content file: skip if already on
    # disk (from an earlier plate in this same loop, or a prior run of this
    # script) rather than tracking "seen" names in memory -- simpler, avoids
    # bash 4+-only associative arrays, and doubles as resumability across runs.
    platemap_dir="${OUT}/platemaps/${DATASET}/${experiment}/platemap"
    platemap_dest="${platemap_dir}/${PLATE_MAP_NAME}.txt"
    if [[ -s "$platemap_dest" ]]; then
        echo "Platemap content already present, skipping: ${platemap_dest}"
    else
        mkdir -p "$platemap_dir"
        aws s3 cp \
          "${BASE}/workspace/metadata/platemaps/${experiment}/platemap/${PLATE_MAP_NAME}.txt" \
          "$platemap_dest" \
          $AWS_OPTS
    fi

    # Resolve the full acquisition folder name so load_data.csv is stored
    # under acquisition_id, not the bare barcode -- the same barcode can
    # occur at multiple timepoints/experiments (see download_images.slurm
    # and datasets/build_metadata_table.py for why). The S3 source path
    # itself is barcode-only; only the local path preserves acquisition_id.
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
    load_data_dir="${OUT}/load_data_csv/${experiment}/${acquisition_id}"
    mkdir -p "$load_data_dir"

    aws s3 cp \
      "${BASE}/workspace/load_data_csv/${experiment}/${plate}/load_data.csv" \
      "${load_data_dir}/load_data.csv" \
      $AWS_OPTS

    echo "Done: ${experiment}/${plate}"
    echo "End: $(date)"
done

echo "Metadata complete"
