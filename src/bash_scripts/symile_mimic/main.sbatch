#!/bin/bash

#SBATCH --job-name=main_symile_wd2_lr0005_clip_logits
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=4:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src

echo 'training...'
python main.py \
    --experiment symile_mimic \
    --batch_sz_train 280 \
    --batch_sz_val 280 \
    --batch_sz_test 280 \
    --check_val_every_n_epoch 1 \
    --d 8192 \
    --drop_last True \
    --epochs 80 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -7 \
    --use_seed True \
    --seed 0 \
    --data_dir /gpfs/scratch/as16583/symile/src/datasets/symile_mimic \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/symile_mimic \
    --pretrained False \
    --negative_sampling n_squared \
    --loss_fn symile \
    --ckpt_path None \
    --wandb False \
    --weight_decay 0.2 \
    --lr 0.0005

echo 'done!'