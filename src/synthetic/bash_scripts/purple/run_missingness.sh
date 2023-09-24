#!/bin/bash

#SBATCH --job-name=pretrain_symile
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=12:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3

echo 'running missingness experiments...'
python main.py \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --check_val_every_n_epoch 1 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/synthetic/ \
    --epochs 2 \
    --logit_scale_init -0.3 \
    --loss_fn symile \
    --lr 1.0e-1 \
    --normalize True \
    --seed 0 \
    --use_seed True \
    --use_full_dataset True \
    --wandb_run_name None \
    --wandb False \
    --concat_infonce True \
    --evaluation support \
    --use_logit_scale_eval True \
    --missingness True \
    --missingness_test True \
    --missingness_p 0.5

echo 'done!'