#!/bin/bash -l
#SBATCH --job-name=prep-phase2
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/prep_phase2_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/prep_phase2_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
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

echo "Starting phase 2 compound aggregation..."

python -u scripts/prepare_phase2_compounds.py "$@"

echo "Phase 2 compound aggregation complete."
