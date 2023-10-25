#!/bin/bash

#SBATCH --job-name=test_zeroshot_symile
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

# NOTE: only 1 GPU should be used when running test.py!

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3

echo 'zeroshot classification...'
python test.py \
    --batch_sz_test 164 \
    --ckpt_path /gpfs/scratch/as16583/ckpts/20231001_193313/epoch=37-val_loss=7.86.ckpt \
    --data_dir_flags /gpfs/scratch/as16583/flags \
    --data_dir_generated_audio /gpfs/scratch/as16583/audio \
    --data_dir_imagenet /gpfs/data/ranganathlab/adriel/imagenet \
    --drop_last False \
    --evaluation zeroshot \
    --limit_test_batches 1.0 \
    --use_precomputed_representations True \
    --precomputed_rep_dir /gpfs/scratch/as16583/dataset \
    --wandb False

echo 'done!'