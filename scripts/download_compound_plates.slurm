#!/bin/bash -l
#SBATCH --job-name=cp-download
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/download_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/download_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=short

# Usage (SLURM passes everything after the script straight through to it).
# Selects a diversity-maximizing set of --target-plates (default 24) compound
# plates from analysis/outputs/compound_plates.csv -- run
# scripts/scan_compound_plates.py first to (re)generate that inventory.
#   sbatch scripts/download_compound_plates.slurm --dry-run
#   sbatch scripts/download_compound_plates.slurm
#   sbatch scripts/download_compound_plates.slurm --target-plates 24 --datasets cpg0000-jump-pilot

REPO=/home/creighton.jo/Cellpainting-SSL-MoA

mkdir -p /scratch/creighton.jo/cellpainting/logs

cd $REPO
export PYTHONPATH=$REPO
export CP_OUTPUT_ROOT=/scratch/creighton.jo/cellpainting

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

echo "Python used:"
which python
python --version
echo "AWS used:"
which aws
aws --version

echo "Starting compound plate download..."

python -u scripts/download_compound_plates.py "$@"
RC=$?

echo "Download script finished with exit code $RC."
exit $RC
