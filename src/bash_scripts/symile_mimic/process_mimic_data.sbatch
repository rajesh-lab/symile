#!/bin/bash

#SBATCH --job-name=process_mimic_data
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/data_processing/symile_mimic

python process_mimic_data.py \
    --mimiciv_hosp_dir /gpfs/data/ranganathlab/mimiciv/2.2/hosp \
    --cxr_data_dir /gpfs/data/ranganathlab/mimic-cxr-jpg/mimic-cxr-jpg-2.0.0.physionet.org \
    --ecg_data_dir /gpfs/data/ranganathlab/mimic-iv-ecg/1.0 \
    --save_dir /gpfs/scratch/as16583/symile/src/datasets/symile_mimic

echo 'done!'