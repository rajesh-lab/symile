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
    --batch_sz_test 300 \
    --ckpt_path /gpfs/scratch/as16583/ckpts/20230922_224922/epoch=40-val_loss=4.62.ckpt \
    --data_dir_flags /gpfs/scratch/as16583/flags \
    --data_dir_generated_audio /gpfs/scratch/as16583/audio \
    --data_dir_imagenet /gpfs/data/ranganathlab/adriel/imagenet \
    --evaluation zeroshot \
    --limit_test_batches 1.0 \
    --wandb False \
    --zeroshot_dataset_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/pretrain_val.csv

echo 'done!'