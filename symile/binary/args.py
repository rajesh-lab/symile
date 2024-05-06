import argparse
from pathlib import Path

from symile.utils import str_to_bool


def parse_args_informations():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_v", type=int, default=2,
                        help="Dimensionality of binary vectors.")

    parser.add_argument("--save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/ckpts/binary"),
                        help="Where to save information results.")

    return parser.parse_args()


def parse_args():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--train_n", type=int, default=100,
                        help="Number of samples (a, b, c) in train dataset.")
    parser.add_argument("--val_n", type=int, default=1000,
                        help="Number of samples (a, b, c) in val dataset.")
    parser.add_argument("--test_n", type=int, default=100,
                        help="Number of samples (a, b, c) in test dataset.")
    parser.add_argument("--d_v", type=int, default=3,
                        help="Dimensionality of dataset vectors.")
    parser.add_argument("--d_r", type=int, default=16,
                        help="Dimensionality of representation vectors.")

    ### TRAINING ARGS ###
    parser.add_argument("--bsz_train", type=int, default=100,
                        help="Batch size for pretraining.")
    parser.add_argument("--bsz_val", type=int, default=1000,
                        help="Val set batch size for pretraining.")
    parser.add_argument("--bsz_test", type=int, default=100,
                        help="Test set batch size.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="Check val every n train epochs.")
    parser.add_argument("--efficient_loss", type=str_to_bool, default=False,
                        help="Whether to compute logits with only \
                              (batch_size^2 - batch_size) negatives.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to train for.")
    parser.add_argument("--logit_scale_init", type=float, default=-0.3,
                        help="Value used to initialize the learned logit_scale. \
                              CLIP used np.log(1 / 0.07) = 2.65926.")
    parser.add_argument("--lr", type=float, default=1.0e-1,
                        help="Learning rate.")
    parser.add_argument("--save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/ckpts/binary/final"),
                        help="Where to save model checkpoints and results.")
    parser.add_argument("--wandb", type=str_to_bool, default=False,
                        help="Whether to use wandb for logging.")

    ### EVALUATION ARGS ###
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()