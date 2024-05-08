#!/bin/bash

#SBATCH --job-name=collect_test_results
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim/data_processing

python collect_test_results.py \
    --results_pt /gpfs/scratch/as16583/symile/symile/cxr_prediction/results/test.yaml \
    --save_pt /gpfs/scratch/as16583/symile/symile/cxr_prediction/results/test.csv

echo 'done!'