import argparse

from sklearn.model_selection import train_test_split
import torch.nn.functional as F


##############
# DATA UTILS #
##############


def split_size(ss):
    """
    Ensure that split size is valid (>= 0.0 and a whole number if > 1.0), and
    convert to integer if > 1.0 (required for sklearn's train_test_split).
    """
    assert ss >= 0.0, "split size must be >= 0.0."
    if ss > 1.0:
        assert ss.is_integer(), "split size must be a whole number if > 1.0."
        ss = int(ss)
    return ss


def get_splits(df, train_size, val_size):
    """
    Split dataframe into train, val, and test sets. Size of the test set is
    inferred from the train and val sizes.

    Args:
        df (pd.DataFrame): Dataframe to split.
        train_size (float): Must be between 0.0 and 1.0. Represents the
                            proportion of the pretrain test words to include
                            in the support classification train split.
        val_size (float): Must be between 0.0 and 1.0. Represents the
                          proportion of the pretrain test words to include
                          in the support classification val split.
    Returns:
        tuple of pd.DataFrame: train, val, and test splits
    """
    train_size = split_size(train_size)
    val_size = split_size(val_size / (1 - train_size))
    train, val = train_test_split(df, train_size=train_size, shuffle=True)
    val, test = train_test_split(val, train_size=val_size, shuffle=True)

    assert len(train) + len(val) + len(test) == len(df), \
        "train/val/test split sizes must add up to length of df."

    return train, val, test


##################
# TRAINING UTILS #
##################


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