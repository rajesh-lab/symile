#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

singularity exec --nv \
	--overlay /vast/work/public/ml-datasets/imagenet/imagenet-object-localization-challenge.sqf:ro \
	--overlay $SCRATCH/cv-corpus-14.0-2023-06-23.sqf:ro \
    --overlay $SCRATCH/overlay-50G-10M-symile.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	/bin/bash -c "

source /ext3/env.sh
cd $SCRATCH/symile/src/symile_m3/data

# pretrain val
echo 'generating pretrain val data...'
python generate_data.py \
    --audio_save_dir /scratch/as16583/symile/src/symile_m3/data/sources/audio \
    --commonvoice_dir /cv-corpus-14.0-2023-06-23 \
    --commonvoice_split train \
    --image_path /scratch/as16583/symile/src/symile_m3/data/sources/img_pretrain_val.csv \
    --n_per_language 500 \
    --negative_samples False \
    --save_path /scratch/as16583/symile/src/symile_m3/data/sources/pretrain_val.csv \
    --text_path /scratch/as16583/symile/src/symile_m3/data/sources/txt_pretrain_val.csv

echo 'done!'
"