#!/bin/bash

#SBATCH --job-name=save_representations
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim_xor

echo 'saving representations...'
python save_representations.py \
    --audio_model_id openai/whisper-tiny \
    --image_model_id openai/clip-vit-base-patch16 \
    --text_model_id bert-base-multilingual-cased \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --drop_last False

echo 'done!'