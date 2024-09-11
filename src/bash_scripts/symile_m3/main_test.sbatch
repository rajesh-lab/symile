#!/bin/bash

#SBATCH --job-name=main_symile_test_2W_2L
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim

echo 'running evaluation...'
python test.py \
    --text_model_id xlm-roberta-large \
    --batch_sz_test 256 \
    --num_langs 2 \
    --data_dir /gpfs/scratch/as16583/symile/src/datasets/symile_m3/2W_2L \
    --save_dir /gpfs/scratch/as16583/results/symile_m3 \
    --ckpt_path /gpfs/scratch/as16583/ckpts_final/symile_m3/20240420_234318_7972/epoch=15-val_loss=0.7591-val_accuracy=0.9418.ckpt \
    --description "testing can delete" \
    --save_representations False \
    --bootstrap True \
    --experiment symile_m3

echo 'done!'