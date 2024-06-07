#!/bin/bash

#SBATCH --job-name=create_umap_reps
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=200GB
#SBATCH --time=0:30:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim/data_processing

echo 'sourcing (translating) data...'

python create_umap_reps.py \
    --n_datapoints 50 \
    --n_components 3 \
    --metric cosine \
    --rep_dir /gpfs/scratch/as16583/results/high_dim/20240425_185140_7761 \
    --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/results/umap/clip

# symile
# rep_dir = Path("/gpfs/scratch/as16583/results/high_dim/20240425_194201_7151")

# clip
# rep_dir = Path("/gpfs/scratch/as16583/results/high_dim/20240425_185140_7761")

echo 'done!'