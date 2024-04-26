#!/bin/bash

#SBATCH --job-name=main_clip_pretrainedT_wd1p_lr0005
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=5:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/risk_prediction

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/healthcare

echo 'training...'
python main.py \
    --batch_sz_train 256 \
    --batch_sz_val 256 \
    --batch_sz_test 256 \
    --check_val_every_n_epoch 1 \
    --d 1024 \
    --drop_last False \
    --epochs 50 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -7 \
    --use_seed True \
    --data_dir /gpfs/scratch/as16583/symile/symile/risk_prediction/datasets \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/risk_prediction \
    --pretrained True \
    --efficient_loss True \
    --loss_fn clip \
    --load_from_ckpt None \
    --wandb True \
    --weight_decay 1.0 \
    --lr 0.0005

echo 'done!'