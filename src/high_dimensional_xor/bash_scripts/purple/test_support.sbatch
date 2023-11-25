#!/bin/bash

#SBATCH --job-name=test_support_symile
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
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
    --ckpt_path /gpfs/scratch/as16583/ckpts/20230925_123311/epoch=124-val_loss=4.27.ckpt \
    --data_dir_flags /gpfs/scratch/as16583/flags \
    --data_dir_generated_audio /gpfs/scratch/as16583/audio \
    --data_dir_imagenet /gpfs/data/ranganathlab/imagenet \
    --epochs 100 \
    --evaluation support \
    --lr 1.0e-3 \
    --support_train_dataset_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_train_1.csv \
    --support_val_dataset_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_val_1.csv \
    --support_test_dataset_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_test_1.csv \
    --use_precomputed_representations True \
    --precomputed_rep_dir /gpfs/scratch/as16583/tensors_100_classes/support \
    --wandb True

echo 'done!'