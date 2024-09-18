#!/bin/bash

#SBATCH --job-name=main_test_clip_10W_10L
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src

echo 'running test script...'
python test.py \
    --experiment symile_m3 \
    --batch_sz_test 256 \
    --num_langs 10 \
    --bootstrap True \
    --bootstrap_n 10 \
    --data_dir /gpfs/scratch/as16583/symile/src/datasets/symile_m3/10W_10L \
    --save_dir /gpfs/scratch/as16583/symile/src/results/symile_m3/10W_10L \
    --ckpt_path /gpfs/scratch/as16583/ckpts_final/symile_m3/20240428_163214_9359/epoch=5-val_loss=13.9158-val_accuracy=0.0927.ckpt \
    --description "ckpt is the best CLIP val loss, \
                   /gpfs/scratch/as16583/ckpts_final/symile_m3/20240428_163214_9359/epoch=5-val_loss=13.9158-val_accuracy=0.0927.ckpt"

echo 'done!'