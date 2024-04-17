import argparse
from pathlib import Path

from symile.utils import str_to_bool


def parse_args_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--step", type=str, default=None,
                        choices=["train", "test"])

    ### DATASET ARGS ###
    parser.add_argument("--data_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/healthcare/datasets"),
                        help="Directory with dataset csvs.")

    ### MODEL ARGS ###
    parser.add_argument("--d", type=int, default=768,
                        help="Dimensionality used by the linear projection heads \
                              of all three encoders.")

    ### TRAINING ARGS ###
    parser.add_argument("--batch_sz_train", type=int, default=300,
                        help="Train batch size.")
    parser.add_argument("--batch_sz_val", type=int, default=300,
                        help="Val batch size.")
    parser.add_argument("--batch_sz_test", type=int, default=300,
                        help="Test batch size.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="Check val every n train epochs.")
    parser.add_argument("--ckpt_save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/ckpts/high_dim"),
                        help="Where to save model checkpoints.")
    parser.add_argument("--drop_last", type=str_to_bool, default=False,
                        help="Whether to drop the last non-full batch of each \
                              DataLoader worker's dataset replica.")
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
                        choices = ["symile", "clip"], default="symile",
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