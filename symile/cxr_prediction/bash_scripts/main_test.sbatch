#!/bin/bash

#SBATCH --job-name=main_test_clip
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_dev
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=0:30:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/cxr_prediction

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/cxr_prediction

echo 'running test script...'
python test.py \
    --batch_sz_test 256 \
    --data_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets \
    --save_dir /gpfs/scratch/as16583/results/cxr_prediction \
    --load_from_ckpt /gpfs/scratch/as16583/ckpts/cxr_prediction/20240424_040914_4926/epoch=17-val_loss=15.2583.ckpt \
    --bootstrap False \
    --description "ckpt is the best CLIP val loss, \
                   /gpfs/scratch/as16583/ckpts/cxr_prediction/20240424_040914_4926/epoch=17-val_loss=15.2583.ckpt \
                   weight_decay 0.0, \
                   learning_rate 0.001, \
                   https://wandb.ai/rajesh-lab/symile/runs/9b8dmujy"

echo 'done!'