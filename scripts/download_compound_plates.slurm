#!/bin/bash -l
#SBATCH --job-name=cp-download
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/download_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/download_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=short

# Usage (SLURM passes everything after the script straight through to it):
#   sbatch scripts/download_compound_plates.slurm \
#     --datasets cpg0000-jump-pilot \
#     --experiments 2020_11_04_CPJUMP1 \
#     --plates-per-experiment 10

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
