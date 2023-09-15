#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

conda activate symile
cd /gpfs/scratch/as16583/symile/src/symile_m3

echo 'pretraining...'
python pretrain.py --train_dataset_path ./pretrain_train.csv \
                   --val_dataset_path ./pretrain_val.csv \
                   --batch_sz 38 \
                   --check_val_every_n_epoch 10 \
                   --epochs 100 \
                   --logit_scale_init 0 \
                   --loss_fn symile \
                   --lr 1.0e-3 \
                   --profiler simple \
                   --wandb True

echo 'done!'
"