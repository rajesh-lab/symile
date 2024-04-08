#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

singularity exec --nv \
    --overlay $SCRATCH/overlay-50G-10M-symile.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	/bin/bash -c "

source /ext3/env.sh
cd $SCRATCH/symile/src/synthetic

echo 'running synthetic experiment...'
python main.py \
    --pretrain_n 5000 \
    --pretrain_val_n 1000 \
    --test_n 5000 \
    --check_val_every_n_epoch 10 \
    --ckpt_save_dir /scratch/as16583/ckpts/synthetic/ \
    --epochs 160 \
    --logit_scale_init -0.3 \
    --loss_fn pairwise_infonce \
    --lr 1.0e-1 \
    --normalize True \
    --seed 0 \
    --use_seed True \
    --use_full_dataset True \
    --wandb_run_name None \
    --wandb True \
    --concat_infonce True \
    --evaluation support \
    --use_logit_scale_eval True

echo 'done!'
"