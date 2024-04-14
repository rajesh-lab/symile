#!/bin/bash

#SBATCH --job-name=merge_representations_2W_2L
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=3:00:00
#SBATCH --partition=a100_dev

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim/data_processing

python merge_representations.py \
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L \
    --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L/train \
    --num_subdirs 20

echo 'done!'