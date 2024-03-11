#!/bin/bash

#SBATCH --job-name=merge_representations
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/data_processing

python merge_representations.py \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/datasets/t5_10M/train_sub_dfs \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/datasets/t5_10M/train \
    --num_subdirs 20

echo 'done!'