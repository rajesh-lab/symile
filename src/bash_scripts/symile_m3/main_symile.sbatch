#!/bin/bash

#SBATCH --job-name=main_symile_testing
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_short
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim

echo 'training...'
python main.py \
    --audio_model_id openai/whisper-large-v3 \
    --image_model_id openai/clip-vit-large-patch14 \
    --text_model_id xlm-roberta-large \
    --batch_sz_train 256 \
    --batch_sz_val 256 \
    --batch_sz_test 256 \
    --check_val_every_n_epoch 2 \
    --d 8192 \
    --drop_last True \
    --epochs 24 \
    --freeze_logit_scale False \
    --limit_train_batches=1.0 \
    --limit_val_batches=1.0 \
    --logit_scale_init -6 \
    --use_seed True \
    --seed 0 \
    --wandb True \
    --missingness False \
    --missingness_prob 0.5 \
    --ckpt_save_dir /gpfs/scratch/as16583/ckpts/high_dim \
    --data_dir /gpfs/scratch/as16583/symile/src/datasets/symile_m3/2W_2L \
    --num_langs 2 \
    --data_reference /gpfs/scratch/as16583/symile/symile/datasets/symile_m3/data_reference.json \
    --negative_sampling n \
    --loss_fn symile \
    --ckpt_path None \
    --weight_decay 0.0 \
    --lr 0.0001 \
    --experiment symile_m3

echo 'done!'