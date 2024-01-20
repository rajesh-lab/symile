#!/bin/bash

#SBATCH --job-name=generate_data_t5_50cls_period
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim

# echo 'generating data for t1...'
# python generate_data.py \
#     --pretrain_n 200000 \
#     --val_size 0.1 \
#     --test_n 10000 \
#     --data_type disjoint \
#     --text_len 1 \
#     --data_reference /gpfs/scratch/as16583/symile/src/high_dim/data_reference_50.json \
#     --save_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t1_50cls/disjoint \

# echo 'generating data for t2...'
# python generate_data.py \
#     --pretrain_n 200000 \
#     --val_size 0.1 \
#     --test_n 10000 \
#     --data_type disjoint \
#     --text_len 2 \
#     --data_reference /gpfs/scratch/as16583/symile/src/high_dim/data_reference_50.json \
#     --save_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t2_50cls_slash/disjoint \

echo 'generating data for t5...'
python generate_data.py \
    --pretrain_n 200000 \
    --val_size 0.1 \
    --test_n 10000 \
    --data_type disjoint \
    --text_len 5 \
    --data_reference /gpfs/scratch/as16583/symile/src/high_dim/data_reference_50.json \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t5_50cls_period/disjoint \

echo 'done!'