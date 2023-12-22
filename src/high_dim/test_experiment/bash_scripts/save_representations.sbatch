#!/bin/bash

#SBATCH --job-name=save_representations_5
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/test_experiment

echo 'saving representations...'
python save_representations.py \
    --audio_model_id openai/whisper-tiny \
    --text_model_id bert-base-multilingual-cased \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --drop_last False \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/c5_l5_t5_50k/disjoint \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/test_experiment/data/c5_l5_t5_50k/disjoint

echo 'done!'