#!/bin/bash

#SBATCH --job-name=pretrain_pairwise_infonce
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3

echo 'pretraining...'
python pretrain.py \
    --audio_model_id openai/whisper-small \
    --image_model_id openai/clip-vit-base-patch16 \
    --text_model_id xlm-roberta-base \
    --batch_sz 300 \
    --batch_sz_val 150 \
    --check_val_every_n_epoch 1 \
    --d 768 \
    --drop_last False \
    --early_stopping False \
    --epochs 5000 \
    --freeze_encoders True \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init 2.65926 \
    --loss_fn pairwise_infonce \
    --lr 1.0e-3 \
    --normalize True \
    --weight_decay 0.01 \
    --wandb True \
    --use_precomputed_representations True \
    --precomputed_rep_dir /gpfs/scratch/as16583/dataset

echo 'done!'