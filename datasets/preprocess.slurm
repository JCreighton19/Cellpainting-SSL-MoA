#!/bin/bash -l
#SBATCH --job-name=prep-cp
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

source activate cellpainting

export CP_OUTPUT_ROOT=/scratch/creighton.jo/cellpainting

python datasets/build_preprocessed_dataset.py