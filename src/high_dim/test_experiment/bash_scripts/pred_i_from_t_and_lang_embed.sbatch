#!/bin/bash

#SBATCH --job-name=pred_i_from_t_lang_embed_t5_50cls
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/test_experiment

echo 'training...'
python pred_i_from_t_and_lang_embed.py \
    --batch_sz_train 300 \
    --batch_sz_val 300 \
    --batch_sz_test 300 \
    --check_val_every_n_epoch 5 \
    --d 768 \
    --drop_last True \
    --epochs 150 \
    --hidden_layer_d 64 \
    --lang_embed_d 16 \
    --lr 1.0e-3 \
    --text_model_id bert-base-multilingual-cased \
    --use_seed True \
    --wandb True \
    --use_precomputed_representations True \
    --concat False \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data/t5_50cls/disjoint \
    --precomputed_rep_dir /gpfs/scratch/as16583/symile/src/high_dim/test_experiment/data/t5_50cls/disjoint \

echo 'done!'