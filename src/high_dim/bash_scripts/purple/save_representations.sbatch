#!/bin/bash

#SBATCH --job-name=save_representations_t5
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=24:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/data_processing

python save_representations.py \
    --audio_model_id openai/whisper-tiny \
    --image_model_id openai/clip-vit-base-patch16 \
    --text_model_id bert-base-multilingual-cased \
    --batch_sz_train 256 \
    --batch_sz_val 256 \
    --batch_sz_test 256 \
    --drop_last False \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/datasets/t5/disjoint \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/datasets/t5/disjoint \

echo 'done!'