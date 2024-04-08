#!/bin/bash

#SBATCH --job-name=max_token_len_t5_10L_10M_overlap
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=4:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/data_processing

python max_token_len.py \
    --text_model_id xlm-roberta-large \
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/test \
    --save_pt /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/test/max_token_len.json

echo 'done!'