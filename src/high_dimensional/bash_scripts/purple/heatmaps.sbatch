#!/bin/bash

#SBATCH --job-name=test_support_symile
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=1:00:00

# NOTE: only 1 GPU should be used when running test.py!

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/dataset/

echo 'heatmaps...'
python heatmaps.py

echo 'done!'