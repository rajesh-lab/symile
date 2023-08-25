import argparse
from utils import str_to_bool


def parse_args():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--pretrain_n", type=int, default=5000,
                        help="Number of samples (a, b, c) in pretraining dataset.")
    parser.add_argument("--pretrain_val_n", type=int, default=1000,
                        help="Number of samples (a, b, c) in pretraining validation dataset.")
    parser.add_argument("--test_n", type=int, default=5000,
                        help="Number of samples (a, b, c) in test dataset.")
    parser.add_argument("--d_v", type=int, default=2,
                        help="Dimensionality of dataset vectors.")
    parser.add_argument("--d_r", type=int, default=2,
                        help="Dimensionality of representation vectors.")

    ### TRAINING ARGS ###
    parser.add_argument("--batch_sz_pt", type=int, default=5000,
                        help="Batch size for pretraining.")
    parser.add_argument("--early_stopping_patience_pt", type=int, default=4,
                        help="Number of times val loss must improve in a row before stopping pretraining.")
    parser.add_argument("--hardcode_encoders", type=str_to_bool, default=False,
                        help="Whether to hardcode encoders during training.")
    parser.add_argument("--logit_scale_init", type=float, default=-0.3,
                        help="Value used to initialize the learned logit_scale. \
                              CLIP used np.log(1 / 0.07) = 2.65926.")
    parser.add_argument("--loss_fn", type=str,
                        choices = ["symile", "pairwise_infonce"], default="pairwise_infonce",
                        help="Loss function to use for training.")
    parser.add_argument("--lr", type=float, default=1.0e-1,
                        help="Learning rate.")
    parser.add_argument("--normalize", type=str_to_bool, default=True,
                        help="Whether to normalize representations, both during \
                              pre-training before loss calculation and during evaluation.")
    parser.add_argument("--pt_epochs", type=int, default=160,
                        help="Number of epochs to pretrain for.")
    parser.add_argument("--pt_val_epochs", type=int, default=10,
                        help="Number of epochs after which to calculate pretraining val loss.")
    parser.add_argument("--use_full_dataset", type=str_to_bool, default=True,
                        help="Whether to set batch size equal to full dataset. \
                              Note that if set to True, `batch_sz` param below \
                              will be ignored.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")
    parser.add_argument("--wandb", type=str_to_bool, default=True,
                        help="Whether to use wandb for logging.")

    ### EVALUATION ARGS ###
    parser.add_argument("--concat_infonce", type=str_to_bool, default=True,
                        help="Whether or not to concatenate (r_a * r_b), (r_b * r_c), (r_a * r_c) for \
                              downstream classification tasks when loss function is 'pairwise_infonce' \
                              (alternative is to sum the three terms).")
    parser.add_argument("--evaluation", type=str,
                        choices=["zeroshot_clf", "support_clf", "sum_clf"],
                        default="support_clf",
                        help="Evaluation method to run.")
    parser.add_argument("--use_logit_scale_eval", type=str_to_bool, default=True,
                        help="Whether or not to scale logits by temperature \
                              parameter during evaluation.")

    return parser.parse_args()