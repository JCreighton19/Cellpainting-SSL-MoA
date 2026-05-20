#!/bin/bash -l
#SBATCH --job-name=umap
#SBATCH --output=logs/umap_%j.out
#SBATCH --error=logs/umap_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

cd ~/Cellpainting-SSL-MoA
export PYTHONPATH=$PWD

python analysis/run_umap.py