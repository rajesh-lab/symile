#!/bin/bash

#SBATCH --job-name=zeroshot
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3

echo 'pretraining...'
CUDA_LAUNCH_BLOCKING=1 python test.py \

echo 'done!'