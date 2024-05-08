#!/bin/bash

#SBATCH --job-name=save_dataset_tensors_val_test
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=4:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction_age_sex/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/cxr_prediction_age_sex/data_processing

python save_dataset_tensors.py \
    --data_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction_age_sex/datasets \
    --cxr_scale 320 \
    --cxr_crop 320 \
    --split_to_run val_test

echo 'done!'