#!/bin/bash

#SBATCH --job-name=main_symile_lr0001
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_short
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=20:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/healthcare

echo 'training...'
python main.py \
    --batch_sz_train 256 \
    --check_val_every_n_epoch 2 \
    --d 8192 \
    --drop_last False \
    --epochs 4 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -7 \
    --lr 0.0001 \
    --use_seed True \
    --wandb True \
    --weight_decay 0.0 \
    --data_dir /gpfs/scratch/as16583/symile/src/healthcare/datasets \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/healthcare \
    --efficient_loss True \
    --loss_fn symile \
    --load_from_ckpt None \
    --labs_model ftt

echo 'done!'