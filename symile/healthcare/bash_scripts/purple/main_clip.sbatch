#!/bin/bash

#SBATCH --job-name=main_clip_lr001
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
    --drop_last True \
    --early_stopping False \
    --early_stopping_patience 20 \
    --epochs 50 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -7 \
    --lr 0.001 \
    --use_seed True \
    --wandb True \
    --weight_decay 0.0 \
    --data_dir /gpfs/scratch/as16583/symile/src/healthcare/datasets \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/healthcare \
    --train_csv train_label.csv \
    --val_csv val_label.csv \
    --zeroshot_task label \
    --efficient_loss True \
    --loss_fn clip \
    --load_from_ckpt None

echo 'done!'