#!/bin/bash

#SBATCH --job-name=get_mimic_data_labs_before_discharge
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=1:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/healthcare/data_processing

python get_mimic_data.py \
    --data_type labs \
    --cxr_type post_24 \
    --ecg_labs_type before_discharge \
    --save_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/cxr_post_ecg_labs

echo 'done!'