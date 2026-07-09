#!/bin/bash -l
#SBATCH --job-name=web-thumbs
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/web_thumbs_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/web_thumbs_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=short

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

echo "Starting web thumbnail generation..."

python -u scripts/generate_web_thumbnails.py "$@"

echo "Thumbnail generation complete."
