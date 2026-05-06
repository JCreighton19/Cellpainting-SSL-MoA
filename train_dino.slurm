#!/bin/bash
#SBATCH --job-name=dino-cp
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

# make sure logs directory exists
mkdir -p logs

# activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpaint

# go to repo root
cd ~/Cellpainting-SSL-MoA

# run training
python train_dino.py