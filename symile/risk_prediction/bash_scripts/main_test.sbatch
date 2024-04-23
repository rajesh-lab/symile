#!/bin/bash

#SBATCH --job-name=main_clip_best_val_acc_at_1
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=0:30:00

# when in AS's BP
source ~/miniconda3/bin/activate symile
cd /gpfs/scratch/as16583/symile/symile/risk_prediction

# when in AP's BP
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate symile
# cd /gpfs/scratch/pulia01/adriel/symile/symile/risk_prediction

echo 'running test script...'
python test.py \
    --batch_sz_test 256 \
    --data_dir /gpfs/scratch/as16583/symile/symile/risk_prediction/datasets \
    --save_dir /gpfs/scratch/as16583/results/risk_prediction \
    --load_from_ckpt /gpfs/scratch/as16583/ckpts/risk_prediction/20240421_233259_8398/epoch=37-val_loss=23.1582.ckpt \
    --bootstrap False \
    --description "ckpt is the best CLIP val acc, \
                   ckpt_pt /gpfs/scratch/as16583/ckpts/risk_prediction/20240421_233259_8398/epoch=37-val_loss=23.1582.ckpt \
                   weight_decay 0.1, \
                   learning_rate 0.001, \
                   https://wandb.ai/rajesh-lab/symile/runs/b245r6xy"

echo 'done!'