import argparse
try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_n", type=int, default=5,
                        help="Number of samples (a, b, c) in pretraining dataset.")
    parser.add_argument("--finetune_n", type=int, default=5000,
                        help="Number of samples (a, b, c) in finetuning dataset.")
    parser.add_argument("--test_n", type=int, default=5000,
                        help="Number of samples (a, b, c) in test dataset.")
    parser.add_argument("--d_v", type=int, default=200,
                        help="Dimensionality of dataset vectors.")
    parser.add_argument("--d_r", type=int, default=2,
                        help="Dimensionality of representation vectors.")
    parser.add_argument("--batch_sz", type=int, default=5000,
                        help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=5.0e-4,
                        help="Learning rate.")
    parser.add_argument("--loss_fn", type=str, default="pairwise_infonce",
                        help="Loss function to use for training. Options are 'symile' and 'pairwise_infonce'.")
    parser.add_argument("--pretrain_epochs", type=int, default=32,
                        help="Number of epochs to pretrain for.")
    parser.add_argument("--finetune_epochs", type=int, default=32,
                        help="Number of epochs to finetune for.")
    parser.add_argument("--normalize", type=bool, default=False,
                        help="Whether to normalize representations before loss calculation.")
    parser.add_argument("--wandb", type=bool, default=False,
                        help="Whether to use wandb for logging.")
    return parser.parse_args()

def wandb_init(args):
    if args.wandb:
        wandb.init(project="symile",
                   config=args)
    return