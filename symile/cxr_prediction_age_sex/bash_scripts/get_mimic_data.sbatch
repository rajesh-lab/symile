#!/bin/bash

#SBATCH --job-name=get_mimic_data_pa_ap
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=50GB
#SBATCH --time=4:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction_age_sex/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/cxr_prediction_age_sex/data_processing

python get_mimic_data.py \
    --save_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction_age_sex/datasets \
    --pa_ap_only True

echo 'done!'