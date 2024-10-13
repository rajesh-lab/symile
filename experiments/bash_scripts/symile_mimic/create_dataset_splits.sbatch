#!/bin/bash

#SBATCH --job-name=create_dataset_splits
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=1:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/data_processing/symile_mimic

python create_dataset_splits.py \
    --dataset_path /gpfs/scratch/as16583/symile/src/datasets/symile_mimic/symile_mimic_data.csv \
    --train_n 10000 \
    --val_n 750 \
    --candidate_n 10 \
    --save_dir /gpfs/scratch/as16583/symile/src/datasets/symile_mimic \
    --use_seed True \
    --seed 0

echo 'done!'