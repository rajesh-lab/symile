#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3/data

echo 'generating data...'
python generate_data.py \
    --pretrain_n 20000 \
    --pretrain_val_size 0.2 \
    --support_n 10000 \
    --support_train_size 0.7 \
    --support_val_size 0.1 \
    --zeroshot_n 4000

echo 'done!'