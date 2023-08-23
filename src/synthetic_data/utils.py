import argparse
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
try:
    import wandb
except ImportError:
    wandb = None


def l2_normalize(vectors):
    """
    L2 normalize a list of 2D vectors.

    Args:
        vectors (list): list of 2D torch.Tensor vectors.
    Returns:
        list of same 2D torch.Tensor vectors, normalized.
    """
    return [F.normalize(v, p=2.0, dim=1) for v in vectors]


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg (str): String representing a boolean.

    Returns:
        Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def wandb_init(args):
    if args.wandb:
        wandb.init(project="symile",
                   config=args)
    return