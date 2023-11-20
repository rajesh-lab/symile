#!/bin/bash

#SBATCH --job-name=create_splits
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

echo 'creating image splits...'
python create_image_splits.py

echo 'creating word splits...'
python create_word_splits.py

echo 'done!'
"