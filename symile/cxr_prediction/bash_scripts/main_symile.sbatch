#!/bin/bash

#SBATCH --job-name=main_symile_seed_9
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=3:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/cxr_prediction

echo 'training...'
python main.py \
    --batch_sz_train 256 \
    --batch_sz_val 256 \
    --batch_sz_test 256 \
    --check_val_every_n_epoch 1 \
    --d 1024 \
    --drop_last False \
    --epochs 100 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -7 \
    --data_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/cxr_24_to_72 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/cxr_prediction \
    --pretrained False \
    --labs_model mlp \
    --efficient_loss True \
    --loss_fn symile \
    --load_from_ckpt None \
    --wandb True \
    --weight_decay 0.1 \
    --lr 0.01 \
    --use_seed True \
    --seed 9

echo 'done!'