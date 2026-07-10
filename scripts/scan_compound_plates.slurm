#!/bin/bash -l
#SBATCH --job-name=cp-gallery-scan
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/gallery_scan_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/gallery_scan_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --partition=short

# Read-only Cell Painting Gallery inventory scan -- does NOT download any
# images and does NOT save any per-plate files permanently. See
# scripts/scan_compound_plates.py for details.
#
# Usage:
#   sbatch scripts/scan_compound_plates.slurm
#   sbatch scripts/scan_compound_plates.slurm --datasets cpg0000-jump-pilot cpg0004-lincs

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

echo "Starting Cell Painting Gallery compound-plate scan..."

python -u scripts/scan_compound_plates.py "$@"
RC=$?

echo "Scan script finished with exit code $RC."
exit $RC
