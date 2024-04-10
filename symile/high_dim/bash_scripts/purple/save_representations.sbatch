#!/bin/bash

#SBATCH --job-name=save_representations_2W_2L_disjoint_train7
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --partition=a100_short

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim/data_processing

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim/data_processing

python save_representations.py \
    --audio_model_id openai/whisper-large-v3 \
    --image_model_id openai/clip-vit-large-patch14 \
    --text_model_id xlm-roberta-large \
    --batch_sz_train 128 \
    --batch_sz_val 128 \
    --batch_sz_test 128 \
    --drop_last False \
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L \
    --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L \
    --max_token_len_pt /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L/max_token_len.json \
    --train_csv train7.csv \
    --split_to_run train

# python save_representations.py \
#     --audio_model_id openai/whisper-large-v3 \
#     --image_model_id openai/clip-vit-large-patch14 \
#     --text_model_id xlm-roberta-large \
#     --batch_sz_train 128 \
#     --batch_sz_val 128 \
#     --batch_sz_test 128 \
#     --drop_last False \
#     --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L \
#     --save_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L \
#     --max_token_len_pt /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L/max_token_len.json \
#     --split_to_run val_test

echo 'done!'