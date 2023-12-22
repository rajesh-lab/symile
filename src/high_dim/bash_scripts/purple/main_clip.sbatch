#!/bin/bash

#SBATCH --job-name=main_50k_clip
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim

# echo 'training clip c2_l2_t2_50k efficient...'
# python main.py \
#     --audio_model_id openai/whisper-tiny \
#     --image_model_id openai/clip-vit-base-patch16 \
#     --text_model_id bert-base-multilingual-cased \
#     --batch_sz_train 300 \
#     --batch_sz_val 300 \
#     --batch_sz_test 300 \
#     --check_val_every_n_epoch 5 \
#     --d 768 \
#     --drop_last True \
#     --early_stopping True \
#     --early_stopping_patience 20 \
#     --epochs 100 \
#     --freeze_encoders True \
#     --freeze_logit_scale False \
#     --limit_train_batches=1.0 \
#     --limit_val_batches=1.0 \
#     --logit_scale_init 0 \
#     --lr 1.0e-3 \
#     --normalize True \
#     --save_test_heatmap True \
#     --use_seed True \
#     --wandb True \
#     --use_precomputed_representations True \
#     --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c2_l2_t2_50k/disjoint \
#     --n_classes 2 \
#     --precomputed_rep_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c2_l2_t2_50k/disjoint \
#     --efficient_loss True \
#     --loss_fn clip \

echo 'training clip c3_l3_t3_50k efficient...'
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
    --epochs 100 \
    --freeze_encoders True \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init 0 \
    --lr 1.0e-3 \
    --normalize True \
    --save_test_heatmap True \
    --use_seed True \
    --wandb True \
    --use_precomputed_representations True \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c3_l3_t3_50k/disjoint \
    --n_classes 3 \
    --precomputed_rep_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c3_l3_t3_50k/disjoint \
    --efficient_loss True \
    --loss_fn clip \

echo 'training clip c4_l4_t4_50k efficient...'
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
    --epochs 100 \
    --freeze_encoders True \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init 0 \
    --lr 1.0e-3 \
    --normalize True \
    --save_test_heatmap True \
    --use_seed True \
    --wandb True \
    --use_precomputed_representations True \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c4_l4_t4_50k/disjoint \
    --n_classes 4 \
    --precomputed_rep_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c4_l4_t4_50k/disjoint \
    --efficient_loss True \
    --loss_fn clip \

echo 'training clip c5_l5_t5_50k efficient...'
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
    --epochs 100 \
    --freeze_encoders True \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init 0 \
    --lr 1.0e-3 \
    --normalize True \
    --save_test_heatmap True \
    --use_seed True \
    --wandb True \
    --use_precomputed_representations True \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c5_l5_t5_50k/disjoint \
    --n_classes 5 \
    --precomputed_rep_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c5_l5_t5_50k/disjoint \
    --efficient_loss True \
    --loss_fn clip \

echo 'done!'