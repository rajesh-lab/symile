#!/bin/bash

#SBATCH --job-name=save_representations_t5_audio_50cls
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=8:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/test_experiment

echo 'saving representations...'
python save_representations_audio.py \
    --audio_model_id openai/whisper-tiny \
    --batch_sz_train 1 \
    --batch_sz_val 1 \
    --batch_sz_test 1 \
    --drop_last False \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t5_50cls/disjoint \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t5_50cls/disjoint

echo 'done!'