#!/bin/bash

#SBATCH --job-name=generate_data_zeroshot
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3/data

echo 'generating zeroshot test data...'
python generate_data.py \
    --audio_save_dir /gpfs/scratch/as16583/symile/src/symile_m3/data/sources \
    --commonvoice_dir /gpfs/data/ranganathlab/adriel/cv/cv-corpus-14.0-2023-06-23 \
    --commonvoice_split test \
    --image_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/img_zeroshot_test.csv \
    --n_per_language 250 \
    --negative_samples False \
    --save_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/zeroshot_test.csv \
    --text_path /gpfs/scratch/as16583/symile/src/symile_m3/data/sources/txt_zeroshot_test.csv

echo 'done!'