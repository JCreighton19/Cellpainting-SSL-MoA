#!/bin/bash
#SBATCH --job-name=dino-cp
#SBATCH --output=logs/dino_%j.out
#SBATCH --error=logs/dino_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module load python/3.13.5

source ~/miniconda3/bin/activate your_env  # if applicable

cd ~/Cellpainting-SSL-MoA

python train_dino.py \
    --data_dir /scratch/$USER/cellpainting/plate1 \
    --batch_size 16 \
    --epochs 1 \
    --num_workers 8