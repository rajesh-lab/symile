#!/bin/bash

#SBATCH --job-name=heatmaps_clip
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_short
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=4:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim

echo 'training...'
python heatmaps.py \
    --audio_model_id openai/whisper-large-v3 \
    --image_model_id openai/clip-vit-large-patch14 \
    --text_model_id xlm-roberta-large \
    --batch_sz_test 256 \
    --d 8192 \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/datasets/t5_10M \
    --data_reference /gpfs/scratch/as16583/symile/src/high_dim/datasets/data_reference.json \
    --load_from_ckpt /gpfs/scratch/as16583/ckpts/high_dim/20240324_193421/epoch=11-val_loss=12.79.ckpt \
    --save_test_heatmaps True \
    --seed 0

echo 'done!'