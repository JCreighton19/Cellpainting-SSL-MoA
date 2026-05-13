#!/bin/bash -l
#SBATCH --job-name=metadata
#SBATCH --output=logs/metadata_%j.out
#SBATCH --error=logs/metadata_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

source /shared/EL9/explorer/anaconda3/2024.06/etc/profile.d/conda.sh
conda activate cellpainting

cd ~/Cellpainting-SSL-MoA
export PYTHONPATH=$PWD

python datasets/build_metadata_table.py