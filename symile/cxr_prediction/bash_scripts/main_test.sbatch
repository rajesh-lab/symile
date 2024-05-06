#!/bin/bash

#SBATCH --job-name=main_test_symile
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
    --data_dir /gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/cxr_24_to_72 \
    --save_dir /gpfs/scratch/as16583/results/cxr_prediction \
    --load_from_ckpt /gpfs/scratch/as16583/ckpts/cxr_prediction/20240427_235312_1971/epoch=39-val_loss=4.6411.ckpt \
    --bootstrap True \
    --bootstrap_n 3 \
    --description "ckpt is the best SYMILE val acc, \
                   /gpfs/scratch/as16583/ckpts/cxr_prediction/20240427_235312_1971/epoch=39-val_loss=4.6411.ckpt \
                   https://wandb.ai/rajesh-lab/symile/runs/uld2rmme"

echo 'done!'