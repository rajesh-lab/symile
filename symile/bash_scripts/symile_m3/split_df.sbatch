#!/bin/bash

#SBATCH --job-name=split_df_2W_2L_disjoint
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --time=4:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim/data_processing

python split_df.py \
    --csv_to_split /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L/train.csv \
    --sub_df_size 500000 \
    --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L

echo 'done!'