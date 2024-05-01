#!/bin/bash

#SBATCH --job-name=main_symile_2W_2L_save_rep
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=4:00:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/high_dim

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/high_dim

echo 'running evaluation...'
python test.py \
    --text_model_id xlm-roberta-large \
    --batch_sz_test 256 \
    --num_langs 2 \
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L \
    --save_dir /gpfs/scratch/as16583/results/high_dim \
    --load_from_ckpt /gpfs/scratch/as16583/ckpts/high_dim/20240420_234318_7972/epoch=23-val_loss=0.7384-val_accuracy=0.9405.ckpt \
    --bootstrap False \
    --description "ckpt is the best SYMILE val loss, saving representations, \
                   ckpt_pt /gpfs/scratch/as16583/ckpts/high_dim/20240420_234318_7972/epoch=23-val_loss=0.7384-val_accuracy=0.9405.ckpt \
                   https://wandb.ai/rajesh-lab/symile/runs/eheyszc4" \
    --save_representations True

echo 'done!'