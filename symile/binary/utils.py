from itertools import product

import numpy as np


def get_vector_support(d):
    """
    Generate all possible values for a binary vector with dimension d.
    """
    binary_combinations = product([0, 1], repeat=d)
    return [np.array(c) for c in binary_combinations]