#!/bin/bash

#SBATCH --job-name=run_zeroshot_d5_efficient
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=8:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/binary

echo 'running d_v = 5 experiment...'
python main.py \
    --batch_sz 1000 \
    --batch_sz_val 1000 \
    --batch_sz_test 1000 \
    --check_val_every_n_epoch 10 \
    --d_r 16 \
    --d_v 5 \
    --efficient_loss True \
    --epochs 100 \
    --pretrain_n 10000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --wandb False \

echo 'done!'