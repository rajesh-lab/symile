#!/bin/bash

#SBATCH --job-name=generate_data_t5_10L_10M_overlap
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim/data_processing

echo 'generating data for t5...'
python generate_data.py \
    --dataset_n 10000000 \
    --val_n 100000 \
    --test_n 100000 \
    --data_type disjoint \
    --num_words 5 \
    --num_langs 5 \
    --data_reference /gpfs/scratch/as16583/symile/symile/high_dim/datasets/data_reference.json \
    --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/test

echo 'done!'