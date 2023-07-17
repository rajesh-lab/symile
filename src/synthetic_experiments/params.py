import argparse
try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_n", type=int, default=10000,
                        help="Number of samples (a, b, c) in pretraining dataset.")
    parser.add_argument("--pretrain_val_n", type=int, default=1000,
                        help="Number of samples (a, b, c) in pretraining validation dataset.")
    parser.add_argument("--early_stopping_patience_pt", type=int, default=4,
                        help="Number of times val loss must improve in a row before stopping pretraining.")
    parser.add_argument("--finetune_n", type=int, default=10000,
                        help="Number of samples (a, b, c) in finetuning dataset.")
    parser.add_argument("--test_n", type=int, default=1000,
                        help="Number of samples (a, b, c) in test dataset.")
    parser.add_argument("--d_v", type=int, default=20,
                        help="Dimensionality of dataset vectors.")
    parser.add_argument("--d_r", type=int, default=10,
                        help="Dimensionality of representation vectors.")
    parser.add_argument("--eps_param", type=float, default=0.01,
                        help="When modulo example is used, eps_param is a multiplier \
                              when generating data: C = ((A+B) % 1) + eps * eps_param.")
    parser.add_argument("--pred_from_reps", type=bool, default=True,
                        help="Whether to predict C_bin from representations r_a and r_b \
                             (as opposed to from A and B directly).")
    parser.add_argument("--use_full_dataset", type=bool, default=True,
                        help="Whether to set batch size equal to full dataset. \
                              Note that if set to True, `batch_sz` param below \
                              will be ignored.")
    parser.add_argument("--example_mod", type=bool, default=False,
                        help="Whether to use modulo example (as opposed to xor example).")
    parser.add_argument("--batch_sz_pt", type=int, default=5000,
                        help="Batch size for pretraining.")
    parser.add_argument("--lr", type=float, default=1.0e-2,
                        help="Learning rate.")
    parser.add_argument("--loss_fn", type=str, default="symile",
                        help="Loss function to use for training. Options are 'symile' and 'pairwise_infonce'.")
    parser.add_argument("--pt_epochs", type=int, default=120,
                        help="Number of epochs to pretrain for.")
    parser.add_argument("--pt_val_epochs", type=int, default=20,
                        help="Number of epochs after which to calculate pretraining val loss.")
    parser.add_argument("--normalize", type=bool, default=True,
                        help="Whether to normalize representations before loss calculation.")
    parser.add_argument("--wandb", type=bool, default=False,
                        help="Whether to use wandb for logging.")
    return parser.parse_args()

def wandb_init(args):
    if args.wandb:
        wandb.init(project="symile",
                   config=args)
    return