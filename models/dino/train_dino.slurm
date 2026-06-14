#!/bin/bash -l
#SBATCH --job-name=dino-cp
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/dino_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/dino_%j.err
#SBATCH --time=08:00:00
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

# Tell dataset where data lives
export CP_DATA_ROOT=$DATA
export CP_OUTPUT_ROOT=$OUT
export HF_HOME=/scratch/creighton.jo/hf_cache
export TORCH_HOME=/scratch/creighton.jo/torch_cache

# Conda setup
source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

PYTHON=$(which python)

echo "Python used:"
$PYTHON -c "import sys; print(sys.executable)"
$PYTHON --version

echo "Torch check:"
$PYTHON -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

echo "Data root:"
echo $CP_DATA_ROOT

echo "Starting training..."

# Resume from previous job if one exists
CURRENT_RUN=$(cat $OUT/checkpoints/current_run 2>/dev/null)
if [ -n "$CURRENT_RUN" ]; then
    export RESUME_DIR=$CURRENT_RUN
    echo "Resuming from: $RESUME_DIR"
else
    echo "Starting fresh run"
fi

$PYTHON models/dino/train_dino.py
EXIT_CODE=$?

# Resubmit if killed before finishing (wall time); do NOT resubmit on clean exit
if [ $EXIT_CODE -ne 0 ]; then
    echo "Job interrupted (exit $EXIT_CODE) — resubmitting..."
    sbatch "$0"
else
    echo "Training complete — not resubmitting."
fi