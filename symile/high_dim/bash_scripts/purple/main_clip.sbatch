#!/bin/bash

#SBATCH --job-name=main_clip_t5_10L_10M_overlap_wd0_lr0001
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
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
    --early_stopping False \
    --early_stopping_patience 20 \
    --epochs 15 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -6 \
    --lr 0.0001 \
    --use_seed True \
    --wandb True \
    --weight_decay 0.0 \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/datasets/t5_10L_10M_overlap \
    --data_reference /gpfs/scratch/as16583/symile/src/high_dim/datasets/data_reference.json \
    --efficient_loss True \
    --loss_fn clip \
    --load_from_ckpt None

echo 'done!'