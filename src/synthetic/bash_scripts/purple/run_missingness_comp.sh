#!/bin/bash

#SBATCH --job-name=missingness_pairwise_infonce_zeroshot
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/synthetic

LOSS="symile"
EVAL="zeroshot"

echo "running '$LOSS' '$EVAL' missingness experiments across different p where there is ALSO missingness in test set..."

echo 'running p = 0.0...'
python main.py \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --check_val_every_n_epoch 10 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/synthetic/ \
    --enable_progress_bar False \
    --epochs 160 \
    --logit_scale_init -0.3 \
    --loss_fn $LOSS \
    --lr 1.0e-1 \
    --normalize True \
    --seed 0 \
    --use_seed True \
    --use_full_dataset True \
    --wandb_run_name None \
    --wandb False \
    --concat_infonce True \
    --evaluation $EVAL \
    --use_logit_scale_eval True \
    --missingness True \
    --missingness_test True \
    --missingness_p 0.0

echo 'running p = 0.2...'
python main.py \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --check_val_every_n_epoch 10 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/synthetic/ \
    --enable_progress_bar False \
    --epochs 160 \
    --logit_scale_init -0.3 \
    --loss_fn $LOSS \
    --lr 1.0e-1 \
    --normalize True \
    --seed 0 \
    --use_seed True \
    --use_full_dataset True \
    --wandb_run_name None \
    --wandb False \
    --concat_infonce True \
    --evaluation $EVAL \
    --use_logit_scale_eval True \
    --missingness True \
    --missingness_test True \
    --missingness_p 0.2

echo 'running p = 0.4...'
python main.py \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --check_val_every_n_epoch 10 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/synthetic/ \
    --enable_progress_bar False \
    --epochs 160 \
    --logit_scale_init -0.3 \
    --loss_fn $LOSS \
    --lr 1.0e-1 \
    --normalize True \
    --seed 0 \
    --use_seed True \
    --use_full_dataset True \
    --wandb_run_name None \
    --wandb False \
    --concat_infonce True \
    --evaluation $EVAL \
    --use_logit_scale_eval True \
    --missingness True \
    --missingness_test True \
    --missingness_p 0.4

echo 'running p = 0.6...'
python main.py \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --check_val_every_n_epoch 10 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/synthetic/ \
    --enable_progress_bar False \
    --epochs 160 \
    --logit_scale_init -0.3 \
    --loss_fn $LOSS \
    --lr 1.0e-1 \
    --normalize True \
    --seed 0 \
    --use_seed True \
    --use_full_dataset True \
    --wandb_run_name None \
    --wandb False \
    --concat_infonce True \
    --evaluation $EVAL \
    --use_logit_scale_eval True \
    --missingness True \
    --missingness_test True \
    --missingness_p 0.6

echo 'running p = 0.8...'
python main.py \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --check_val_every_n_epoch 10 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/synthetic/ \
    --enable_progress_bar False \
    --epochs 160 \
    --logit_scale_init -0.3 \
    --loss_fn $LOSS \
    --lr 1.0e-1 \
    --normalize True \
    --seed 0 \
    --use_seed True \
    --use_full_dataset True \
    --wandb_run_name None \
    --wandb False \
    --concat_infonce True \
    --evaluation $EVAL \
    --use_logit_scale_eval True \
    --missingness True \
    --missingness_test True \
    --missingness_p 0.8

echo 'running p = 1.0...'
python main.py \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --check_val_every_n_epoch 10 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/synthetic/ \
    --enable_progress_bar False \
    --epochs 160 \
    --logit_scale_init -0.3 \
    --loss_fn $LOSS \
    --lr 1.0e-1 \
    --normalize True \
    --seed 0 \
    --use_seed True \
    --use_full_dataset True \
    --wandb_run_name None \
    --wandb False \
    --concat_infonce True \
    --evaluation $EVAL \
    --use_logit_scale_eval True \
    --missingness True \
    --missingness_test True \
    --missingness_p 1.0

echo 'done!'