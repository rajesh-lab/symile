#!/bin/bash

#SBATCH --job-name=test_experiment_disjoint_c5_l5_t5_50k
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/test_experiment

echo 'training...'
python test_experiment.py \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/test_experiment/data/c5_l5_t5_50k/disjoint \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --check_val_every_n_epoch 5 \
    --d 768 \
    --drop_last True \
    --epochs 50 \
    --hidden_layer_d 64 \
    --lang_embed_d 16 \
    --lr 1.0e-3 \
    --audio_model_id openai/whisper-tiny \
    --text_model_id bert-base-multilingual-cased \
    --use_seed True \
    --wandb False \
    --use_precomputed_representations True \
    --precomputed_rep_dir /gpfs/scratch/as16583/symile/src/high_dim/test_experiment/data/c5_l5_t5_50k/disjoint

echo 'done!'