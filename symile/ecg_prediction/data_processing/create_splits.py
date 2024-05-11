import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from args import parse_create_splits


def split_train_val(df, seed):
    unique_subject_ids = df["subject_id"].unique()
    train_subject_ids, val_subject_ids = train_test_split(
        unique_subject_ids, test_size=0.10, shuffle=True, random_state=seed)

    train_df = df[df["subject_id"].isin(train_subject_ids)]
    val_df = df[df["subject_id"].isin(val_subject_ids)]

    return train_df, val_df


def assert_no_patient_overlap(train_df, val_df, test_df):
    assert set(train_df["subject_id"]).isdisjoint(set(val_df["subject_id"]))
    assert set(train_df["subject_id"]).isdisjoint(set(test_df["subject_id"]))
    assert set(val_df["subject_id"]).isdisjoint(set(test_df["subject_id"]))


if __name__ == '__main__':
    start = time.time()

    args = parse_create_splits()

    df = pd.read_csv(args.dataset_path)

    # get splits ensuring that there is no patient overlap
    unique_subject_ids = df["subject_id"].unique()
    train_subject_ids, val_subject_ids = train_test_split(unique_subject_ids, train_size=0.8, shuffle=True, random_state=args.seed)
    val_subject_ids, test_subject_ids = train_test_split(val_subject_ids, test_size=0.5, shuffle=True, random_state=args.seed)

    train_df = df[df["subject_id"].isin(train_subject_ids)]
    val_df = df[df["subject_id"].isin(val_subject_ids)]
    test_df = df[df["subject_id"].isin(test_subject_ids)]

    assert_no_patient_overlap(train_df, val_df, test_df)

    # create age quantiles using the train set, and then apply them to the val and test sets
    train_df["age_quantile"], train_bins = pd.qcut(
        train_df["age"], q=4, labels=False, retbins=True, duplicates="raise")

    val_df["age_quantile"], val_bins = pd.cut(
        val_df["age"], bins=train_bins, labels=False, retbins=True, include_lowest=True)

    test_df["age_quantile"], test_bins = pd.cut(
        test_df["age"], bins=train_bins, labels=False, retbins=True, include_lowest=True)

    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    test_df.to_csv(args.save_dir / "test.csv", index=False)

    with open(args.save_dir / "metadata.json", "w") as f:
        json.dump({
            "train set size": len(train_df),
            "val set size": len(val_df),
            "test set size": len(test_df)
        }, f, indent=4)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")