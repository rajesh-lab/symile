#!/bin/bash

#SBATCH --job-name=add_missingness_2W_2L
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=0:30:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim/data_processing

python add_missingness.py \
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L_missingness \
    --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L_missingness \
    --missingness_prob 0.5

echo 'done!'