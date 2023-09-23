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

echo 'pretraining...'
python pretrain.py \
    --batch_sz 250 \
    --batch_sz_val 250 \
    --check_val_every_n_epoch 1 \
    --epochs 3 \
    --freeze_encoders True \
    --limit_train_batches=0.25 \
    --limit_val_batches=0.25 \
    --logit_scale_init 0 \
    --loss_fn symile \
    --lr 1.0e-3 \
    --normalize True \
    --weight_decay 0.1 \
    --wandb False \
    --use_precomputed_representations False \
    --precomputed_rep_dir /gpfs/scratch/as16583/random_tensors

echo 'done!'