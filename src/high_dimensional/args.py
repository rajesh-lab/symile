import argparse
from pathlib import Path

from src.utils import str_to_bool


def parse_args_pretrain():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--data_dir_flags", type=Path,
                        default=Path("/gpfs/scratch/as16583/flags"),
                        help="Directory with flag image files.")
    parser.add_argument("--data_dir_generated_audio", type=Path,
                        default=Path("/gpfs/scratch/as16583/audio"),
                        help="Directory with generated audio files.")
    parser.add_argument("--data_dir_imagenet", type=Path,
                        default=Path("/gpfs/data/ranganathlab/imagenet"),
                        help="Directory with ImageNet data files.")
    parser.add_argument("--train_dataset_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/pretrain_train.csv"),
                        help="Path to train dataset csv.")
    parser.add_argument("--val_dataset_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/pretrain_val.csv"),
                        help="Path to val dataset csv.")

    ### MODEL ARGS ###
    parser.add_argument("--audio_model_id", type=str,
                        default="openai/whisper-tiny",
                        choices = ["openai/whisper-small", "openai/whisper-tiny"],
                        help="Hugging Face model id for audio encoder.")
    parser.add_argument("--image_model_id", type=str,
                        default="openai/clip-vit-base-patch16",
                        help="Hugging Face model id for image encoder.")
    parser.add_argument("--text_model_id", type=str,
                        default="bert-base-multilingual-cased",
                        choices = ["bert-base-multilingual-cased", "xlm-roberta-base"],
                        help="Hugging Face model id for text encoder.")
    parser.add_argument("--d", type=int, default=768,
                        help="Dimensionality used by the linear projection heads \
                              of all three encoders.")
    parser.add_argument("--text_embedding", type=str,
                        choices = ["eos", "bos"], default="eos",
                        help="Whether to use text encoder BOS or EOS embedding \
                              as input to projection head.")

    ### TRAINING ARGS ###
    parser.add_argument("--batch_sz", type=int, default=300,
                        help="Batch size for pretraining.")
    parser.add_argument("--batch_sz_val", type=int, default=300,
                        help="Val set batch size for pretraining.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5,
                        help="Check val every n train epochs.")
    parser.add_argument("--ckpt_save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/ckpts"),
                        help="Where to save model checkpoints.")
    parser.add_argument("--drop_last", type=str_to_bool, default=True,
                        help="Whether to drop the last non-full batch of each \
                              DataLoader worker's dataset replica.")
    parser.add_argument("--early_stopping", type=str_to_bool, default=True,
                        help="Whether to use early stopping.")
    parser.add_argument("--early_stopping_patience", type=int, default=20,
                        help="Number of val checks with no improvement after \
                              which pre-training will be stopped.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to pretrain for.")
    parser.add_argument("--freeze_encoders", type=str_to_bool, default=True,
                        help="Whether to freeze encoders during pretraining.")
    parser.add_argument("--freeze_logit_scale", type=str_to_bool, default=False,
                        help="Whether to freeze logit scale during pretraining.")
    parser.add_argument("--logit_scale_init", type=float, default=0,
                        help="Value used to initialize the learned logit_scale. \
                              CLIP used np.log(1 / 0.07) = 2.65926.")
    parser.add_argument("--loss_fn", type=str,
                        choices = ["symile", "pairwise_infonce"], default="symile",
                        help="Loss function to use for training.")
    parser.add_argument("--lr", type=float, default=1.0e-3,
                        help="Learning rate.")
    parser.add_argument("--normalize", type=str_to_bool, default=True,
                        help="Whether to normalize representations, both during \
                              pre-training before loss calculation and during evaluation.")
    parser.add_argument("--profiler", type=str,
                        choices=["none", "simple", "advanced"],
                        default="none",
                        help="Profiler to use to find bottlenecks in code.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")
    parser.add_argument("--wandb", type=str_to_bool, default=True,
                        help="Whether to use wandb for logging.")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay coefficient used by AdamW optimizer.")

    ### DEBUGGING ARGS ###
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="How much of training dataset to check. Useful \
                              when debugging. 1.0 is default used by Trainer. \
                              Set to 0.1 to check 10% of dataset.")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="How much of val dataset to check. Useful \
                              when debugging. 1.0 is default used by Trainer. \
                              Set to 0.1 to check 10% of dataset.")
    parser.add_argument("--use_precomputed_representations", type=str_to_bool, default=True,
                        help="Whether to use precomputed representations to \
                              train projection heads.")
    parser.add_argument("--precomputed_rep_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/tensors_100_classes"),
                        help="Where precomputed representations are saved.")

    return parser.parse_args()


def parse_args_test():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--data_dir_flags", type=Path,
                        default=Path("/gpfs/scratch/as16583/flags"),
                        help="Directory with flag image files.")
    parser.add_argument("--data_dir_generated_audio", type=Path,
                        default=Path("/gpfs/scratch/as16583/audio"),
                        help="Directory with generated audio files.")
    parser.add_argument("--data_dir_imagenet", type=Path,
                        default=Path("/gpfs/data/ranganathlab/imagenet"),
                        help="Directory with ImageNet data files.")
    parser.add_argument("--support_train_dataset_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_train.csv"),
                        help="Path to support classification train (finetune) \
                        dataset csv.")
    parser.add_argument("--support_val_dataset_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_val.csv"),
                        help="Path to support classification val dataset csv.")
    parser.add_argument("--support_test_dataset_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/support_test.csv"),
                        help="Path to support classification test dataset csv.")
    parser.add_argument("--zeroshot_dataset_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/zeroshot_test.csv"),
                        help="Path to zeroshot classification dataset csv.")

    ### MODEL ARGS ###
    parser.add_argument("--ckpt_path", type=Path,
                        default=Path("/scratch/as16583/symile/src/symile_data/ckpts/pretrain/20230908_231703/epoch=0-val_loss=2.30.ckpt"),
                        help="Path to pretrained encoders checkpoint.")

    ### TRAINING ARGS ###
    parser.add_argument("--batch_sz", type=int, default=2,
                        help="Batch size for training support classification.")
    parser.add_argument("--batch_sz_val", type=int, default=300,
                        help="Val set batch size for training support classification.")
    parser.add_argument("--batch_sz_test", type=int, default=300,
                        help="Test set batch size for support or zeroshot classification.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="Check val every n train epochs.")
    parser.add_argument("--ckpt_save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/ckpts"),
                        help="Where to save model checkpoints.")
    parser.add_argument("--drop_last", type=str_to_bool, default=True,
                        help="Whether to drop the last non-full batch of each \
                              DataLoader worker's dataset replica.")
    parser.add_argument("--early_stopping_patience", type=int, default=4,
                        help="Number of val checks with no improvement after \
                              which training will be stopped.")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs to train for.")
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="How much of training dataset to check. Useful \
                              when debugging. 1.0 is default used by Trainer. \
                              Set to 0.1 to check 10% of dataset.")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="How much of val dataset to check. Useful \
                              when debugging. 1.0 is default used by Trainer. \
                              Set to 0.1 to check 10% of dataset.")
    parser.add_argument("--limit_test_batches", type=float, default=1.0,
                        help="How much of test dataset to check. Useful \
                              when debugging. 1.0 is default used by Trainer. \
                              Set to 0.1 to check 10% of dataset.")
    parser.add_argument("--lr", type=float, default=1.0e-1,
                        help="Learning rate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")
    parser.add_argument("--wandb", type=str_to_bool, default=False,
                        help="Whether to use wandb for logging.")

    ### EVALUATION ARGS ###
    parser.add_argument("--concat_infonce", type=str_to_bool, default=False,
                        help="Whether or not to concatenate (r_a * r_b), (r_b * r_c), (r_a * r_c) for \
                              downstream classification tasks when loss function is 'pairwise_infonce' \
                              (alternative is to sum the three terms).")
    parser.add_argument("--evaluation", type=str,
                        choices=["zeroshot", "support"],
                        default="zeroshot",
                        help="Evaluation method to run.")
    parser.add_argument("--use_logit_scale", type=str_to_bool, default=True,
                        help="Whether or not to scale logits by temperature \
                              parameter.")

    ### DEBUGGING ARGS ###
    parser.add_argument("--use_precomputed_representations", type=str_to_bool, default=True,
                        help="Whether to use precomputed representations.")
    parser.add_argument("--precomputed_rep_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/tensors_100_classes"),
                        help="Where precomputed representations are saved.")

    return parser.parse_args()