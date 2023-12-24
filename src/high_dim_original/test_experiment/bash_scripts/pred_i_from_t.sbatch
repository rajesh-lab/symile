#!/bin/bash

#SBATCH --job-name=pred_i_from_t
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --time=6:00:00

source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/src/high_dim/test_experiment

echo 'training...'
python pred_i_from_t.py \
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
    --text_model_id bert-base-multilingual-cased \
    --use_seed True \
    --wandb True \
    --use_precomputed_representations True \
    --data_dir /gpfs/scratch/as16583/symile/src/high_dim/data_c5_l5_new/disjoint \
    --precomputed_rep_dir /gpfs/scratch/as16583/symile/src/high_dim/test_experiment/data_c5_l5_new/disjoint

echo 'done!'