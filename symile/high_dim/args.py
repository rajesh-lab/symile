import argparse
from pathlib import Path

from symile.utils import str_to_bool


def parse_args_main():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")
    parser.add_argument("--train_csv", type=Path,
                        default=Path("train.csv"),
                        help="Filename for train csv.")
    parser.add_argument("--val_csv", type=Path,
                        default=Path("val.csv"),
                        help="Filename for val csv.")
    parser.add_argument("--test_csv", type=Path,
                        default=Path("test.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--data_reference", type=Path,
                        help="Path to json file with class names, ImageNet synset \
                              id, and language translations.")
    parser.add_argument("--num_langs", type=int,
                        help="Number of languages in generated text.")

    ### MODEL ARGS ###
    parser.add_argument("--audio_model_id", type=str,
                        help="Hugging Face model id for audio encoder.")
    parser.add_argument("--image_model_id", type=str,
                        help="Hugging Face model id for image encoder.")
    parser.add_argument("--text_model_id", type=str,
                        help="Hugging Face model id for text encoder.")
    parser.add_argument("--d", type=int,
                        help="Dimensionality used by the linear projection heads \
                              of all three encoders.")
    parser.add_argument("--text_embedding", type=str,
                        choices = ["eos", "bos"], default="eos",
                        help="Whether to use text encoder BOS or EOS embedding \
                              as input to projection head.")
    parser.add_argument("--metadata_filename", type=Path,
                        default=Path("metadata.json"),
                        help="json filename with metadata for all encoders.")

    ### TRAINING ARGS ###
    parser.add_argument("--batch_sz_train", type=int, default=256,
                        help="Train batch size for pretraining.")
    parser.add_argument("--batch_sz_val", type=int, default=256,
                        help="Val set batch size for pretraining.")
    parser.add_argument("--batch_sz_test", type=int, default=256,
                        help="Test set batch size.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=2,
                        help="Check val every n train epochs.")
    parser.add_argument("--ckpt_save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/ckpts/high_dim"),
                        help="Where to save model checkpoints.")
    parser.add_argument("--drop_last", type=str_to_bool, default=True,
                        help="Whether to drop the last non-full batch of each \
                              DataLoader worker's dataset replica.")
    parser.add_argument("--early_stopping", type=str_to_bool, default=True,
                        help="Whether to use early stopping.")
    parser.add_argument("--early_stopping_patience", type=int, default=20,
                        help="Number of val checks with no improvement after \
                              which pre-training will be stopped.")
    parser.add_argument("--efficient_loss", type=str_to_bool, default=True,
                        help="Whether to compute logits with only \
                              (batch_size^2 - batch_size) negatives.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to pretrain for.")
    parser.add_argument("--freeze_logit_scale", type=str_to_bool, default=False,
                        help="Whether to freeze logit scale during pretraining.")
    parser.add_argument("--load_from_ckpt", type=str,
                        default=None,
                        help="Checkpoint to load from.")
    parser.add_argument("--logit_scale_init", type=float, default=0,
                        help="Value used to initialize the learned logit_scale. \
                              CLIP used np.log(1 / 0.07) = 2.65926.")
    parser.add_argument("--loss_fn", type=str,
                        choices = ["symile", "clip"],
                        help="Loss function to use for training.")
    parser.add_argument("--lr", type=float, default=1.0e-3,
                        help="Learning rate.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")
    parser.add_argument("--wandb", type=str_to_bool, default=False,
                        help="Whether to use wandb for logging.")
    parser.add_argument("--wandb_run_id", type=str,
                        default=None,
                        help="Use if loading from checkpoint and using WandbLogger.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
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

    return parser.parse_args()


def parse_args_test():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_from_ckpt", type=str,
                        default=None,
                        help="Checkpoint to load from.")
    parser.add_argument("--save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/results/high_dim"),
                        help="Where to save test results.")

    ### DATA ARGS ###
    parser.add_argument("--batch_sz_test", type=int, default=300,
                        help="Test set batch size.")
    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")
    parser.add_argument("--num_langs", type=int, default=5,
                        help="Number of languages in generated text.")
    parser.add_argument("--text_model_id", type=str,
                        help="Hugging Face model id for text encoder.")

    ### BOOTSTRAP ARGS ###
    parser.add_argument("--bootstrap", type=str_to_bool, default=False,
                        help="Whether to compute a two-sided bootstrap confidence \
                              interval of evaluation metric.")
    parser.add_argument("--bootstrap_n", type=int, default=20,
                        help="Number of resamples performed to form the bootstrap \
                              distribution of evaluation metric.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")

    return parser.parse_args()


def parse_args_collect_tuning_results():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_pt", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/symile/high_dim/results/2W_2L/tuning_runs.yaml"),
                        help="Path to yaml file with hyperparameter tuning results.")
    parser.add_argument("--save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/results/high_dim/results/2W_2L"),
                        help="Where to save test results.")

    return parser.parse_args()