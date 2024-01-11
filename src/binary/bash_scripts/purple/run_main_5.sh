#!/bin/bash

#SBATCH --job-name=run_main_5
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=23:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/binary

python main.py \
    --train_n 10000 \
    --val_n 1000 \
    --test_n 5000 \
    --bsz_train 1000 \
    --bsz_val 1000 \
    --bsz_test 1000 \
    --check_val_every_n_epoch 10 \
    --d_r 16 \
    --d_v 5 \
    --efficient_loss True \
    --epochs 100 \
    --seed 5 \
    --wandb False \

echo 'done!'