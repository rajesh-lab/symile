#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim_xor

echo 'generating data...'
python generate_data.py \
    --pretrain_n 50000 \
    --val_size 0.1 \
    --test_n 10000 \
    --support_train_size 0.8 \
    --support_val_size 0.1

echo 'done!'