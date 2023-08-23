#!/bin/bash

EVALUATION="sum_clf"

# hyperparameters
loss_fn=("symile" "pairwise_infonce")
seed=(0 1)

for i in "${loss_fn[@]}"; do
    for j in "${seed[@]}"; do
        python main.py \
            --logit_scale_init -0.3 \
            --normalize True \
            --use_logit_scale_eval True \
            --lr 0.1 \
            --seed $j \
            --loss_fn $i \
            --evaluation $EVALUATION
    done
done