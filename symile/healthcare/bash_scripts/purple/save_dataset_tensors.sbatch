#!/bin/bash

#SBATCH --job-name=save_dataset_tensors
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=3:00:00
#SBATCH --partition=a100_dev

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/healthcare/data_processing

python save_dataset_tensors.py \
    --data_dir /gpfs/scratch/as16583/symile/src/healthcare/datasets \
    --cxr_scale 320 \
    --cxr_crop 320

echo 'done!'