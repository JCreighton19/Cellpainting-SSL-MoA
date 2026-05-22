#!/bin/bash -l
#SBATCH --job-name=prep-cp
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/creighton.jo/cellpainting/logs/prep_%j.out
#SBATCH --error=/scratch/creighton.jo/cellpainting/logs/prep_%j.err

source activate cellpainting

export CP_OUTPUT_ROOT=/scratch/creighton.jo/cellpainting
export CP_DATA_ROOT=/scratch/creighton.jo/cellpainting/data/raw

python -u datasets/build_preprocessed_dataset.py