#!/bin/bash

#SBATCH --job-name=main_symile_missingness70_val_acc
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --time=1:00:00

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
    --data_dir /gpfs/scratch/as16583/symile/symile/high_dim/datasets/disjoint/2W_2L_missingness \
    --save_dir /gpfs/scratch/as16583/results/high_dim \
    --load_from_ckpt /gpfs/data/ranganathlab/high_dim_ckpts/20240517_022206_0198/epoch=23-val_loss=5.1873-val_acc=0.6377.ckpt \
    --bootstrap False \
    --description "/gpfs/data/ranganathlab/high_dim_ckpts/20240517_022206_0198/epoch=23-val_loss=5.1873-val_acc=0.6377.ckpt \
                   https://wandb.ai/rajesh-lab/symile/runs/0tra2aid" \
    --save_representations False

echo 'done!'