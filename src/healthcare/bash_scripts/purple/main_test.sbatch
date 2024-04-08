#!/bin/bash

#SBATCH --job-name=main_test
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --partition=a100_dev
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150GB
#SBATCH --time=3:00:00

# when in AS's BP
# source ~/miniconda3/bin/activate symile
# cd /gpfs/scratch/as16583/symile/symile/healthcare

# when in AP's BP
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate symile
cd /gpfs/scratch/pulia01/adriel/symile/symile/healthcare

echo 'training...'
python main.py \
    --step test \
    --batch_sz_train 256 \
    --wandb False \
    --ckpt_save_dir /gpfs/scratch/pulia01/adriel/ckpts/healthcare \
    --data_dir /gpfs/scratch/pulia01/adriel/symile/src/healthcare/datasets \
    --load_from_ckpt /gpfs/scratch/pulia01/adriel/ckpts/healthcare/20240403_231215_5743/best_val_loss_epoch=29-val_loss=6.07.ckpt

echo 'done!'