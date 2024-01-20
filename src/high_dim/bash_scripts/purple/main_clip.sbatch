#!/bin/bash

#SBATCH --job-name=main_clip_t5_50cls_lr000005
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim

echo 'training...'
python main.py \
    --audio_model_id openai/whisper-tiny \
    --image_model_id openai/clip-vit-base-patch16 \
    --text_model_id bert-base-multilingual-cased \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --check_val_every_n_epoch 1 \
    --d 768 \
    --drop_last True \
    --early_stopping False \
    --early_stopping_patience 20 \
    --epochs 1000 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init 0 \
    --lr 0.000005 \
    --save_test_heatmap False \
    --use_seed True \
    --wandb True \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t5_50cls/disjoint \
    --data_reference /gpfs/scratch/as16583/symile/src/high_dim/data_reference_50.json \
    --efficient_loss True \
    --loss_fn clip \
    --load_from_ckpt None

echo 'done!'