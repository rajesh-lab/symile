import argparse
try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=str,
                        default="support_clf",
                        help="Evaluation methods to run. Must be comma-separated. \
                              Options are: zeroshot_clf, support_clf, sum_clf.")
    parser.add_argument("--pretrain_n", type=int, default=5000,
                        help="Number of samples (a, b, c) in pretraining dataset.")
    parser.add_argument("--pretrain_val_n", type=int, default=1000,
                        help="Number of samples (a, b, c) in pretraining validation dataset.")
    parser.add_argument("--early_stopping_patience_pt", type=int, default=4,
                        help="Number of times val loss must improve in a row before stopping pretraining.")
    parser.add_argument("--test_n", type=int, default=10,
                        help="Number of samples (a, b, c) in test dataset.")
    parser.add_argument("--d_v", type=int, default=2,
                        help="Dimensionality of dataset vectors.")
    parser.add_argument("--d_r", type=int, default=2,
                        help="Dimensionality of representation vectors.")
    parser.add_argument("--use_full_dataset", type=bool, default=True,
                        help="Whether to set batch size equal to full dataset. \
                              Note that if set to True, `batch_sz` param below \
                              will be ignored.")
    parser.add_argument("--batch_sz_pt", type=int, default=5000,
                        help="Batch size for pretraining.")
    parser.add_argument("--lr", type=float, default=1.0e-2,
                        help="Learning rate.")
    parser.add_argument("--loss_fn", type=str, default="symile",
                        help="Loss function to use for training. \
                              Options are 'symile' and 'pairwise_infonce'.")
    parser.add_argument("--concat_infonce", type=bool, default=False,
                        help="Whether or not to concatenate (r_a * r_b), (r_b * r_c), (r_a * r_c) for \
                              downstream classification tasks when loss function is 'pairwise_infonce' \
                              (alternative is to sum the three terms).")
    parser.add_argument("--use_temp_param_eval", type=bool, default=False,
                        help="Whether or not to scale logits by temperature \
                              parameter during evaluation.")
    parser.add_argument("--pt_epochs", type=int, default=1,
                        help="Number of epochs to pretrain for.")
    parser.add_argument("--pt_val_epochs", type=int, default=10,
                        help="Number of epochs after which to calculate pretraining val loss.")
    parser.add_argument("--normalize", type=bool, default=True,
                        help="Whether to normalize representations before loss calculation.")
    parser.add_argument("--normalize_eval", type=bool, default=False,
                        help="Whether to normalize representations during evaluation.")
    parser.add_argument("--logit_scale_init", type=float, default=2.65926,
                        help="Value used to initialize the learned logit_scale. \
                              CLIP used np.log(1 / 0.07) = 2.65926.")
    parser.add_argument("--hardcode_encoders", type=bool, default=True,
                        help="Whether to hardcode encoders during training.")
    parser.add_argument("--wandb", type=bool, default=False,
                        help="Whether to use wandb for logging.")
    return parser.parse_args()

def wandb_init(args):
    if args.wandb:
        wandb.init(project="symile",
                   config=args)
    return