#!/bin/bash
#SBATCH --job-name=dino-cp
#SBATCH --partition=gpu-short
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpaint

cd ~/Cellpainting-SSL-MoA

python train_dino.py