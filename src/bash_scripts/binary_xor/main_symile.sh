#!/bin/bash

#SBATCH --job-name=testing
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile

python main.py \
    --batch_sz_train 1000 \
    --batch_sz_val 1000 \
    --batch_sz_test 1000 \
    --check_val_every_n_epoch 10 \
    --drop_last True \
    --epochs 100 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -0.3 \
    --use_seed True \
    --wandb False \
    --missingness False \
    --missingness_prob 0.5 \
    --train_n 10000 \
    --val_n 1000 \
    --test_n 5000 \
    --d_r 16 \
    --d_v 5 \
    --negative_sampling n \
    --seed 0 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/binary \
    --loss_fn symile \
    --lr 0.1 \
    --experiment binary_xor

echo 'done!'