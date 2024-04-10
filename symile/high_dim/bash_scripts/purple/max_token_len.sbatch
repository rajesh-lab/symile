#!/bin/bash

#SBATCH --job-name=max_token_len_10W_10L_disjoint
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=4:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim/data_processing

python max_token_len.py \
    --text_model_id xlm-roberta-large \
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/10W_10L \
    --save_pt /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/10W_10L/max_token_len.json

echo 'done!'