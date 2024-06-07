#!/bin/bash

#SBATCH --job-name=save_dataset_tensors
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=4:00:00
#SBATCH --partition=a100_dev

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/healthcare/data_processing

python save_dataset_tensors.py \
    --data_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets \
    --cxr_scale 320 \
    --cxr_crop 320

echo 'done!'