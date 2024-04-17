#!/bin/bash

#SBATCH --job-name=main_symile_mlp_wd1_lr0001
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=2:00:00

# when in AS's BP
# source ~/miniconda3/bin/activate symile
# cd /gpfs/scratch/as16583/symile/symile/healthcare

# when in AP's BP
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate symile
cd /gpfs/scratch/pulia01/adriel/symile/symile/healthcare

echo 'training...'
python main.py \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --check_val_every_n_epoch 2 \
    --d 1024 \
    --drop_last False \
    --epochs 50 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -7 \
    --lr 0.0001 \
    --use_seed True \
    --wandb False \
    --weight_decay 0.1 \
    --data_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/cxr_prediction \
    --efficient_loss True \
    --loss_fn symile \
    --load_from_ckpt None \
    --step train

echo 'done!'