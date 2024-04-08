#!/bin/bash

#SBATCH --job-name=main_test
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_dev
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=3:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/healthcare

echo 'training...'
python main.py \
    --step test \
    --batch_sz_train 256 \
    --wandb True \
    --load_from_ckpt  /gpfs/scratch/as16583/ckpts/healthcare/20240402_125839/ckpts/healthcare/20240402_125839/best_val_loss_epoch=3-val_loss=6.46.ckpt

echo 'done!'