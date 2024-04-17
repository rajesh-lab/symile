#!/bin/bash

#SBATCH --job-name=main_symile_2W_2L_wd001_lr00001
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_short
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim

echo 'training...'
python main.py \
    --audio_model_id openai/whisper-large-v3 \
    --image_model_id openai/clip-vit-large-patch14 \
    --text_model_id xlm-roberta-large \
    --batch_sz_train 256 \
    --batch_sz_val 256 \
    --batch_sz_test 256 \
    --check_val_every_n_epoch 2 \
    --d 8192 \
    --drop_last True \
    --early_stopping True \
    --early_stopping_patience 5 \
    --epochs 25 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -6 \
    --lr 0.00001 \
    --weight_decay 0.001 \
    --use_seed True \
    --wandb True \
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L \
    --num_langs 2 \
    --data_reference /gpfs/scratch/as16583/symile/symile/high_dim/datasets/data_reference.json \
    --efficient_loss True \
    --loss_fn symile \
    --load_from_ckpt /gpfs/scratch/as16583/ckpts/high_dim/20240415_181732_5622/best_val_loss_epoch=23-val_loss=0.74.ckpt \
    --wandb_run_id hfgvpyvs

echo 'done!'