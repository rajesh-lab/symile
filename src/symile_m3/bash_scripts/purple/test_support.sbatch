#!/bin/bash

#SBATCH --job-name=test_support_symile
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

# NOTE: only 1 GPU should be used when running test.py!

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3

echo 'support classification...'
python test.py \
    --batch_sz 250 \
    --batch_sz_val 250 \
    --batch_sz_test 250 \
    --check_val_every_n_epoch 1 \
    --ckpt_path /gpfs/scratch/as16583/symile/src/symile_m3/ckpts/pretrain/20230918_191959/epoch=4-val_loss=5.53.ckpt \
    --data_dir_commonvoice /gpfs/data/ranganathlab/adriel/cv/cv-corpus-14.0-2023-06-23 \
    --data_dir_flags /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/flags \
    --data_dir_generated_audio /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/audio \
    --data_dir_imagenet /gpfs/data/ranganathlab/adriel/imagenet \
    --epochs 100 \
    --evaluation support \
    --lr 1.0e-3 \
    --support_train_dataset_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_train.csv \
    --support_val_dataset_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_val.csv \
    --support_test_dataset_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_test.csv \
    --wandb True

echo 'done!'