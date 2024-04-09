#!/bin/bash

#SBATCH --job-name=main_symile_t5_10L_10M_overlap_wd0_lr0
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_short
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=3:00:00

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
    --num_langs 5 \
    --save_dir /gpfs/scratch/as16583/results/high_dim \
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/test \
    --load_from_ckpt /gpfs/scratch/as16583/ckpts/high_dim/20240324_193322/epoch=13-val_loss=0.73.ckpt \
    --bootstrap True \
    --bootstrap_n 10

echo 'done!'