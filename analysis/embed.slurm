#!/bin/bash -l
#SBATCH --job-name=embed
#SBATCH --output=logs/embed_%j.out
#SBATCH --error=logs/embed_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

cd ~/Cellpainting-SSL-MoA
export PYTHONPATH=$PWD

python analysis/embedding_umap.py