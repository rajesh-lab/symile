#!/bin/bash

#SBATCH --job-name=generate_data_support
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3/data

# support train
echo 'generating support train data...'
python generate_data.py \
    --audio_save_dir /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/audio \
    --commonvoice_dir /gpfs/data/ranganathlab/adriel/cv/cv-corpus-14.0-2023-06-23 \
    --commonvoice_split dev \
    --image_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/img_support_train.csv \
    --n_per_language 1000 \
    --negative_samples True \
    --save_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_train.csv \
    --text_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/txt_support_train.csv

# support val
echo 'generating support val data...'
python generate_data.py \
    --audio_save_dir /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/audio \
    --commonvoice_dir /gpfs/data/ranganathlab/adriel/cv/cv-corpus-14.0-2023-06-23 \
    --commonvoice_split dev \
    --image_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/img_support_val.csv \
    --n_per_language 125 \
    --negative_samples True \
    --save_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_val.csv \
    --text_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/txt_support_val.csv

# support test
echo 'generating support test data...'
python generate_data.py \
    --audio_save_dir /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/audio \
    --commonvoice_dir /gpfs/data/ranganathlab/adriel/cv/cv-corpus-14.0-2023-06-23 \
    --commonvoice_split test \
    --image_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/img_support_test.csv \
    --n_per_language 250 \
    --negative_samples True \
    --save_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_test.csv \
    --text_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/txt_support_test.csv

echo 'done!'