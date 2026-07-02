#!/bin/bash -l
#SBATCH --job-name=dino-cp
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/%x_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --requeue


# Paths
REPO=/home/creighton.jo/Cellpainting-SSL-MoA
DATA=/scratch/creighton.jo/cellpainting/data/raw
OUT=/scratch/creighton.jo/cellpainting

mkdir -p /scratch/creighton.jo/cellpainting/logs

cd $REPO

# Python path
export PYTHONPATH=$REPO

# Dataset + cache paths
export CP_DATA_ROOT=$DATA
export CP_OUTPUT_ROOT=$OUT
export HF_HOME=/scratch/creighton.jo/hf_cache
export TORCH_HOME=/scratch/creighton.jo/torch_cache

# Use pre-generated foreground tiles (optional toggle)
export CP_USE_TILES=0

# Explicit run mode control:
#   resume = continue last run (default)
#   new    = force new experiment
export CP_RUN_MODE=${CP_RUN_MODE:-resume}

# Conda setup
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

PYTHON=$(which python)

echo "Python used:"
$PYTHON -c "import sys; print(sys.executable)"
$PYTHON --version

echo "Torch check:"
$PYTHON -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

echo "Starting training..."
echo "CP_RUN_MODE = $CP_RUN_MODE"

# ----------------------------
# RUN DIRECTORY RESOLUTION
# ----------------------------
RUN_ROOT="$OUT/checkpoints"
CURRENT_RUN_FILE="$RUN_ROOT/current_run"

if [ "$CP_RUN_MODE" == "new" ]; then

    echo "Creating NEW experiment run"

    RESUME_DIR="$RUN_ROOT/$(date +%m%d%y_%H%M%S)"
    mkdir -p "$RESUME_DIR"

    echo "$RESUME_DIR" > "$CURRENT_RUN_FILE"

else
    if [ -f "$CURRENT_RUN_FILE" ]; then
        RESUME_DIR=$(cat "$CURRENT_RUN_FILE")
        echo "Resuming existing run: $RESUME_DIR"
    else
        echo "No existing run found → creating new run"

        RESUME_DIR="$RUN_ROOT/$(date +%m%d%y_%H%M%S)"
        mkdir -p "$RESUME_DIR"

        echo "$RESUME_DIR" > "$CURRENT_RUN_FILE"
    fi
fi

export RESUME_DIR

# ----------------------------
# SINGLE CONTINUOUS LOG FILE
# ----------------------------
LOGFILE="$RESUME_DIR/train.log"
touch "$LOGFILE"

echo "" >> "$LOGFILE"
echo "==================================================" >> "$LOGFILE"
echo "SLURM JOB: $SLURM_JOB_ID" >> "$LOGFILE"
echo "START TIME: $(date)" >> "$LOGFILE"
echo "RUN DIR: $RESUME_DIR" >> "$LOGFILE"
echo "CP_RUN_MODE: $CP_RUN_MODE" >> "$LOGFILE"
echo "==================================================" >> "$LOGFILE"
echo "" >> "$LOGFILE"

# ----------------------------
# TRAINING
# ----------------------------
$PYTHON models/dino/train_dino.py 2>&1 | tee -a "$LOGFILE"
EXIT_CODE=${PIPESTATUS[0]}

# ----------------------------
# RESUBMIT LOGIC
# ----------------------------
if [ $EXIT_CODE -ne 0 ]; then
    echo "Exit code: $EXIT_CODE"
else
    echo "Training complete — not resubmitting."
fi