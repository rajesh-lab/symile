#!/bin/bash

#SBATCH --job-name=generate_data_10W_10L_disjoint
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim/data_processing

echo 'generating data...'
python generate_data.py \
    --train_n 10000000 \
    --val_n 500000 \
    --test_n 500000 \
    --data_type disjoint \
    --num_words 10 \
    --num_langs 10 \
    --data_reference /gpfs/scratch/as16583/symile/symile/high_dim/datasets/data_reference.json \
    --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/10W_10L

echo 'done!'