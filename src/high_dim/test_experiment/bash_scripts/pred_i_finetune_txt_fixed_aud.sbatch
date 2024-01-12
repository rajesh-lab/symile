#!/bin/bash

#SBATCH --job-name=pred_i_finetune_txt_enc_t5_50cls_lr_0001_layer_norm
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/test_experiment

echo 'training...'
python pred_i_finetune_txt_fixed_aud.py \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --check_val_every_n_epoch 1 \
    --d 768 \
    --drop_last True \
    --epochs 50 \
    --hidden_layer_d 64 \
    --use_seed True \
    --wandb True \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t5_50cls/disjoint \
    --num_classes 50 \
    --lr 0.0001 \
    --weight_decay 0.0

echo 'done!'