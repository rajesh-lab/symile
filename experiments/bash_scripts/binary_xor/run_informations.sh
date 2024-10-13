#!/bin/bash

#SBATCH --job-name=run_informations_dv5
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/binary

python informations.py --d_v 5 --save_dir /gpfs/scratch/as16583/ckpts/binary/informations

echo 'done!'