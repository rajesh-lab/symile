#!/bin/bash

#SBATCH --job-name=pretrain_symile
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3

echo 'sanity checking...'
python sanity_check.py \
    --audio_model_id openai/whisper-small \
    --image_model_id openai/clip-vit-base-patch16 \
    --text_model_id xlm-roberta-base \
    --batch_sz 1000 \
    --batch_sz_val 1000 \
    --check_val_every_n_epoch 1 \
    --epochs 200 \
    --freeze_encoders True \
    --freeze_logit_scale True \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init 0 \
    --loss_fn symile \
    --lr 1.0e-3 \
    --normalize True \
    --weight_decay 0.01 \
    --wandb True \
    --use_precomputed_representations True \
    --precomputed_rep_dir /gpfs/scratch/as16583/xlm_whispersmall_tensors/pretrain

echo 'done!'