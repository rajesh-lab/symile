#!/bin/bash

#SBATCH --job-name=pretrain_support
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=4:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/binary

echo 'running experiments...'
python main.py \
    --batch_sz 1000 \
    --batch_sz_val 1000 \
    --batch_sz_test 5000 \
    --check_val_every_n_epoch 10 \
    --concat_infonce True \
    --d_r 16 \
    --d_v 1 \
    --epochs 100 \
    --evaluation support \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --wandb True

echo 'done!'