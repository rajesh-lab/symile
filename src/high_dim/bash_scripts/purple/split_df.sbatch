#!/bin/bash

#SBATCH --job-name=split_df_t5_10L_10M_overlap
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=4:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/data_processing

python split_df.py \
    --csv_to_split /gpfs/scratch/as16583/symile/src/high_dim/datasets/t5_10L_10M_overlap/train.csv \
    --sub_df_size 500000 \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/datasets/t5_10L_10M_overlap

echo 'done!'