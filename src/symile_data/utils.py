"""
TODO:
- probably pull this document out so it's shared with synthetic data
- data get_allowed_license_ids
"""

import argparse
import os
import random

import numpy as np
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

################
# COCO dataset #
################
def get_allowed_license_ids(licenses, noderivs=False):
    """
    explain no derives here (maybe include html?)
    expects a list, returns a list
    """
    allowed_licenses = [
        # http://creativecommons.org/licenses/by-nc-sa/2.0/
        "Attribution-NonCommercial-ShareAlike License",
        # http://creativecommons.org/licenses/by-nc/2.0/
        "Attribution-NonCommercial License",
        # http://creativecommons.org/licenses/by/2.0/
        "Attribution License",
        # http://creativecommons.org/licenses/by-sa/2.0/
        "Attribution-ShareAlike License",
        # http://flickr.com/commons/usage/
        "No known copyright restrictions"
    ]
    allowed_licenses_with_nd = [
        # http://creativecommons.org/licenses/by-nc-nd/2.0/
        "Attribution-NonCommercial-NoDerivs License",
        # http://creativecommons.org/licenses/by-nd/2.0/
        "Attribution-NoDerivs License"
    ]
    allowed_license_ids = []
    for license in licenses:
        if license["name"] in allowed_licenses:
            allowed_license_ids.append(license["id"])
        elif license["name"] in allowed_licenses_with_nd:
            if noderivs:
                allowed_license_ids.append(license["id"])
        else:
            raise NameError('License name is unknown.')
    return allowed_license_ids