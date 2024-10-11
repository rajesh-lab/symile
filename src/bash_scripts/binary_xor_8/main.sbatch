#!/bin/bash

#SBATCH --job-name=main_symile_v_h
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=4:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src

python main.py \
    --experiment binary_xor_8 \
    --batch_sz_train 1000 \
    --batch_sz_val 1000 \
    --batch_sz_test 1000 \
    --check_val_every_n_epoch 1 \
    --drop_last True \
    --epochs 50 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -0.3 \
    --use_seed True \
    --seed 0 \
    --wandb True \
    --train_n 10000 \
    --val_n 1000 \
    --test_n 5000 \
    --d 16 \
    --negative_sampling n \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/binary8 \
    --bootstrap False \
    --bootstrap_n 10 \
    --loss_fn symile \
    --lr 0.1 \
    --weight_decay 0.01

echo 'done!'