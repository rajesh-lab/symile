import argparse
from pathlib import Path

from src.utils import str_to_bool


def parse_args():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--pretrain_n", type=int, default=100,
                        help="Number of samples (a, b, c) in pretraining dataset.")
    parser.add_argument("--pretrain_val_n", type=int, default=1000,
                        help="Number of samples (a, b, c) in pretraining validation dataset.")
    parser.add_argument("--test_n", type=int, default=100,
                        help="Number of samples (a, b, c) in test dataset.")
    parser.add_argument("--d_v", type=int, default=3,
                        help="Dimensionality of dataset vectors.")
    parser.add_argument("--d_r", type=int, default=16,
                        help="Dimensionality of representation vectors.")

    ### TRAINING ARGS ###
    parser.add_argument("--batch_sz", type=int, default=100,
                        help="Batch size for pretraining.")
    parser.add_argument("--batch_sz_val", type=int, default=1000,
                        help="Val set batch size for pretraining.")
    parser.add_argument("--batch_sz_test", type=int, default=100,
                        help="Test set batch size.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="Check val every n train epochs.")
    parser.add_argument("--ckpt_save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/ckpts/binary"),
                        help="Where to save model checkpoints.")
    parser.add_argument("--efficient_loss", type=str_to_bool, default=False,
                        help="Whether to compute logits with only \
                              (batch_size^2 - batch_size) negatives.")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to train for.")
    parser.add_argument("--hardcode_encoders", type=str_to_bool, default=False,
                        help="Whether to hardcode encoders during training.")
    parser.add_argument("--logit_scale_init", type=float, default=-0.3,
                        help="Value used to initialize the learned logit_scale. \
                              CLIP used np.log(1 / 0.07) = 2.65926.")
    parser.add_argument("--lr", type=float, default=1.0e-1,
                        help="Learning rate.")
    parser.add_argument("--normalize", type=str_to_bool, default=True,
                        help="Whether to normalize representations, both during \
                              pre-training before loss calculation and during evaluation.")
    parser.add_argument("--save_likelihood_ratios", type=str_to_bool, default=False,
                        help="Whether to save true likelihood ratio \
                              p(a,b,c)/p(a)p(b)p(c) for each i_p.")
    parser.add_argument("--save_loss_results", type=str_to_bool, default=False,
                        help="Whether to save loss results.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")
    parser.add_argument("--use_full_dataset", type=str_to_bool, default=False,
                        help="Whether to set batch size equal to full dataset. \
                              Note that if set to True, `batch_sz` param \
                              will be ignored.")
    parser.add_argument("--wandb", type=str_to_bool, default=False,
                        help="Whether to use wandb for logging.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Run name for wandb for logging.")

    ### EVALUATION ARGS ###
    parser.add_argument("--enable_progress_bar", type=str_to_bool, default=True,
                        help="Whether to enable or disable the progress bar.")
    parser.add_argument("--use_logit_scale_eval", type=str_to_bool, default=True,
                        help="Whether or not to scale logits by temperature \
                              parameter during evaluation.")

    return parser.parse_args()