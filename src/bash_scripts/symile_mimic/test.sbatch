#!/bin/bash

#SBATCH --job-name=main_test_symile
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_dev
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=1:30:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src

echo 'running test script...'
python test.py \
    --experiment symile_mimic \
    --batch_sz_test 256 \
    --bootstrap True \
    --bootstrap_n 10 \
    --data_dir /gpfs/scratch/as16583/symile/src/datasets/symile_mimic \
    --save_dir /gpfs/scratch/as16583/symile/src/results/symile_mimic \
    --ckpt_path /gpfs/scratch/as16583/ckpts/symile_mimic/20240911_153438_2282/epoch=20-val_loss=9.5965.ckpt \
    --description "ckpt is the best Symile val loss, \
                   /gpfs/scratch/as16583/ckpts/symile_mimic/20240911_153438_2282/epoch=20-val_loss=9.5965.ckpt"

echo 'done!'