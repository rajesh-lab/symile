#!/bin/bash

#SBATCH --job-name=create_splits
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=1:00:00
#SBATCH --partition=a100_dev

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/healthcare/data_processing

python create_splits.py \
    --dataset_path /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/dataset.csv \
    --save_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets

echo 'done!'