from symile.high_dim.constants import *


def get_language_constant(num_langs):
    if num_langs == 10:
        return LANGUAGES_10
    elif num_langs == 5:
        return LANGUAGES_5
    elif num_langs == 2:
        return LANGUAGES_2