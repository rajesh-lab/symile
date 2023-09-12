#!/bin/bash

#SBATCH --job-name=generate_data
#SBATCH --output=./out/%j_%x.out
#SBATCH --error=./out/%j_%x.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=50GB
#SBATCH --time=6:00:00

singularity exec --nv \
	--overlay /vast/work/public/ml-datasets/imagenet/imagenet-object-localization-challenge.sqf:ro \
	--overlay $SCRATCH/cv-corpus-14.0-2023-06-23.sqf:ro \
    --overlay $SCRATCH/overlay-50G-10M-symile.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
	/bin/bash -c "

source /ext3/env.sh
cd $SCRATCH/symile/src/symile_m3/data_sources

# pretrain train
echo 'generating pretrain train data...'
python generate_data.py --commonvoice_split train \
                        --image_path ./img_pretrain_train.csv \
                        --n_per_language 2000 \
                        --negative_samples False \
                        --save_path ./pretrain_train.csv \
                        --text_path ./txt_pretrain_train.csv

# pretrain val
echo 'generating pretrain val data...'
python generate_data.py --commonvoice_split train \
                        --image_path ./img_pretrain_val.csv \
                        --n_per_language 2000 \
                        --negative_samples False \
                        --save_path ./pretrain_val.csv \
                        --text_path ./txt_pretrain_val.csv

# zeroshot test
echo 'generating zeroshot test data...'
python generate_data.py --commonvoice_split test \
                        --image_path ./img_zeroshot_test.csv \
                        --n_per_language 2000 \
                        --negative_samples False \
                        --save_path ./zeroshot_test.csv \
                        --text_path ./txt_zeroshot_test.csv

# support train
echo 'generating support train data...'
python generate_data.py --commonvoice_split dev \
                        --image_path ./img_support_train.csv \
                        --n_per_language 2000 \
                        --negative_samples True \
                        --save_path ./support_train.csv \
                        --text_path ./txt_support_train.csv

# support val
echo 'generating support val data...'
python generate_data.py --commonvoice_split dev \
                        --image_path ./img_support_val.csv \
                        --n_per_language 2000 \
                        --negative_samples True \
                        --save_path ./support_val.csv \
                        --text_path ./txt_support_val.csv

# support test
echo 'generating support test data...'
python generate_data.py --commonvoice_split test \
                        --image_path ./img_support_test.csv \
                        --n_per_language 2000 \
                        --negative_samples True \
                        --save_path ./support_test.csv \
                        --text_path ./txt_support_test.csv

echo 'done!'
"