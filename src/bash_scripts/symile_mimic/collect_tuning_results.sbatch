#!/bin/bash

#SBATCH --job-name=collect_tuning_results
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=10GB
#SBATCH --time=1:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src

python collect_tuning_results.py \
    --results_pt /gpfs/scratch/as16583/symile/src/results/symile_mimic/tuning_runs.yaml \
    --save_pt /gpfs/scratch/as16583/symile/src/results/symile_mimic/tuning_runs.csv \
    --experiment symile_mimic

echo 'done!'