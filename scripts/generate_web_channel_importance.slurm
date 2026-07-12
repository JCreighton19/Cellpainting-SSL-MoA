#!/bin/bash -l
#SBATCH --job-name=web-chanimp
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/web_chanimp_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/web_chanimp_%j.err
#SBATCH --time=04:00:00
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

echo "Starting web channel-importance generation..."

python -u scripts/generate_web_channel_importance.py "$@"

echo "Channel-importance generation complete."
