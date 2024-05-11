#!/bin/bash

#SBATCH --job-name=main_symile_all_negs_pretrainedF_wd01_lr0005
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction_age_sex

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/cxr_prediction_age_sex

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
    --data_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction_age_sex/datasets \
    --ckpt_save_dir /gpfs/data/ranganathlab/symile_tmp/cxr_prediction_age_sex_ckpts \
    --pretrained False \
    --efficient_loss False \
    --loss_fn symile \
    --load_from_ckpt None \
    --wandb True \
    --weight_decay 0.01 \
    --lr 0.0005

echo 'done!'