#!/bin/bash -l
#SBATCH --job-name=prep-phase1
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/prep_phase1_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/prep_phase1_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short

REPO=/home/creighton.jo/Cellpainting-SSL-MoA

mkdir -p /scratch/creighton.jo/cellpainting/logs

cd $REPO
export PYTHONPATH=$REPO

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

echo "Python used:"
which python
python --version

echo "Starting phase 1 data prep..."

python -u scripts/prepare_phase1_data.py "$@"

echo "Phase 1 data prep complete."
