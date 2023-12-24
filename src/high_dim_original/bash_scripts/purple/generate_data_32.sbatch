#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim

echo 'generating data...'

python generate_data.py \
    --pretrain_n 50000 \
    --val_size 0.1 \
    --test_n 10000 \
    --data_type disjoint \
    --n_classes 3 \
    --n_languages 3 \
    --text_len 2 \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c3_l3_t2_50k/disjoint \

python generate_data.py \
    --pretrain_n 50000 \
    --val_size 0.1 \
    --test_n 10000 \
    --data_type disjoint \
    --n_classes 4 \
    --n_languages 4 \
    --text_len 2 \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c4_l4_t2_50k/disjoint \

python generate_data.py \
    --pretrain_n 50000 \
    --val_size 0.1 \
    --test_n 10000 \
    --data_type disjoint \
    --n_classes 5 \
    --n_languages 5 \
    --text_len 2 \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c5_l5_t2_50k/disjoint \

echo 'done!'