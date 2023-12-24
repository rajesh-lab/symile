#!/bin/bash

#SBATCH --job-name=source_data
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim

echo 'sourcing (translating) data...'

python source_data.py \
    --data_reference /gpfs/scratch/as16583/symile/src/high_dim/data_reference.json \
    --imagenet_classmapping_path /gpfs/data/ranganathlab/imagenet/LOC_synset_mapping.txt \

echo 'done!'