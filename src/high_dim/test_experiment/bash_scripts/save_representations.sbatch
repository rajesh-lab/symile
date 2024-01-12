#!/bin/bash

#SBATCH --job-name=save_representations_t5_50cls_attn
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/test_experiment

echo 'saving representations...'
python save_representations_attn.py \
    --audio_model_id openai/whisper-tiny \
    --text_model_id bert-base-multilingual-cased \
    --text_embedding eos \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --drop_last False \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t5_50cls_attn/disjoint \
    --save_dir /gpfs/scratch/as16583/symile/src/high_dim/test_experiment/data/t5_50cls_attn/disjoint \

echo 'done!'