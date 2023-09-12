from sklearn.model_selection import train_test_split


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