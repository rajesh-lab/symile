import argparse
from argparse import Namespace
from itertools import product
from json import JSONEncoder
from pathlib import Path

import numpy as np
import torch.nn.functional as F

from constants import LANGUAGES_2, LANGUAGES_5, LANGUAGES_10


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


def get_vector_support(d):
    """
    Generate all possible values for a binary vector with dimension d.
    """
    binary_combinations = product([0, 1], repeat=d)
    return [np.array(c) for c in binary_combinations]


def l2_normalize(vectors):
    """
    L2 normalize a list of 2D vectors.

    Args:
        vectors (list): list of 2D torch.Tensor vectors.
    Returns:
        list of same 2D torch.Tensor vectors, normalized.
    """
    return [F.normalize(v, p=2.0, dim=1) for v in vectors]


def get_language_constant(num_langs):
    """
    For the Symile-M3 experiments. Returns a list of language abbreviations
    (ISO-639 codes) based on the specified number of languages.
    """
    if num_langs == 10:
        return LANGUAGES_10
    elif num_langs == 5:
        return LANGUAGES_5
    elif num_langs == 2:
        return LANGUAGES_2


class PathToStrEncoder(JSONEncoder):
    """
    Custom JSON encoder that converts Path and Namespace objects to JSON
    serializable formats by overriding the default method to handle Path and
    Namespace objects.
    """
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)     # convert Path object to string
        elif isinstance(obj, Namespace):
            return vars(obj)    # convert Namespace object to dictionary
        return JSONEncoder.default(self, obj)  # default method