#!/bin/bash

#SBATCH --job-name=create_dataset
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=5:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/healthcare/data_processing

python create_dataset.py \
    --dataset_type cxr_post_24 \
    --cxr_df_path /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/cxr_post_24/cxr_df.csv \
    --ecg_df_path /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/cxr_post_24/ecg_df.csv \
    --labs_df_path /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/cxr_post_24/labs_df.csv \
    --save_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/cxr_post_24

echo 'done!'