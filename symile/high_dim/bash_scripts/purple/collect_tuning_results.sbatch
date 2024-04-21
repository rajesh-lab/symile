#!/bin/bash

#SBATCH --job-name=collect_tuning_results
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim/data_processing

echo 'sourcing (translating) data...'

python collect_tuning_results.py \
    --results_pt /gpfs/scratch/as16583/symile/symile/high_dim/results/2W_2L/tuning_runs.yaml \
    --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/results/2W_2L

echo 'done!'