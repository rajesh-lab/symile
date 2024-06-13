#!/bin/bash

#SBATCH --job-name=process_and_save_tensors
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/data_processing/symile_mimic

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/healthcare/data_processing

python process_and_save_tensors.py \
    --data_dir /gpfs/scratch/as16583/symile/src/datasets/symile_mimic \
    --ecg_data_dir /gpfs/data/ranganathlab/mimic-iv-ecg/1.0 \
    --cxr_data_dir /gpfs/data/ranganathlab/mimic-cxr-jpg/mimic-cxr-jpg-2.0.0.physionet.org \
    --cxr_scale 320 \
    --cxr_crop 320

echo 'done!'