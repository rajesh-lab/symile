#!/bin/bash

#SBATCH --job-name=main_symile
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim_xor

echo 'training...'
python main.py \
    --audio_model_id openai/whisper-tiny \
    --image_model_id openai/clip-vit-base-patch16 \
    --text_model_id bert-base-multilingual-cased \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --check_val_every_n_epoch 5 \
    --d 768 \
    --drop_last True \
    --early_stopping True \
    --early_stopping_patience 20 \
    --epochs 50 \
    --freeze_encoders True \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init 0 \
    --loss_fn symile \
    --lr 1.0e-3 \
    --normalize True \
    --use_seed True \
    --wandb True \
    --use_precomputed_representations True \
    --precomputed_rep_dir /gpfs/scratch/as16583/symile/src/high_dim_xor/data

echo 'done!'