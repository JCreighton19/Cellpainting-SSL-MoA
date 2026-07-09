#!/bin/bash -l
#SBATCH --job-name=web-attn
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/web_attn_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/web_attn_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

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

echo "Starting web attention map generation..."

python -u scripts/generate_web_attention_maps.py "$@"

echo "Attention map generation complete."
