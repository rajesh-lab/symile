import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from args import parse_create_splits
from symile.cxr_prediction_age_sex.constants import CHEXPERT_LABELS


def split_query_candidate(df, args, seed):
    """
    We focus on the eight CheXpert labels.

    For each label, we do the following to create the test set:
    - sample 5 query samples (ECG + labs)
    - sample 125 candidate samples (CXRs): 5 positive, 120 negative
    """
    assert args.num_positive >= args.num_query, "The number of positive candidates \
        must be greater than the number of query samples because the positive \
        candidates must contain the true CXR from the query samples."

    # Get the query samples. We ensure that each query sample is positive for
    # exactly_one label.
    query_df = pd.DataFrame()

    for label in CHEXPERT_LABELS:
        # get exclusive positive subset
        other_labels = [l for l in CHEXPERT_LABELS + ["No Finding"] if l != label]
        conditions = (df[label] == 1.0) & (df[other_labels].isin([0.0, np.nan]).all(axis=1))
        assert len(df[conditions]) >= args.num_query, "subset we sample from must be large enough"
        query_samples = df[conditions].sample(n=args.num_query, replace=False, random_state=seed)

        # add to query_df
        query_samples["label_name"] = label
        query_samples["label_value"] = 1.0
        query_df = pd.concat([query_df, query_samples])

    # Get the candidate samples. Candidate samples can be positive for more than
    # one label at a time. We ensure all query samples for each label are among
    # the positive candidate samples.
    candidate_df = pd.DataFrame()

    for label in CHEXPERT_LABELS:

        # get samples from query_df that are positive for this label
        positive_candidates = query_df[query_df.label_name == label]
        assert len(positive_candidates) == args.num_query

        # remove those samples from df to sample other positive samples
        num_remaining_positives = args.num_positive - len(positive_candidates)
        if num_remaining_positives > 0:
            df_filtered = df[~df["subject_id"].isin(positive_candidates.subject_id.to_list())]
            conditions = (df_filtered[label] == 1.0) & (df_filtered["No Finding"].isin([0.0, np.nan]))
            assert len(df_filtered[conditions]) >= num_remaining_positives, "subset we sample from must be large enough"
            remaining_positives = df_filtered[conditions].sample(n=num_remaining_positives, replace=False, random_state=seed)
            positive_candidates = pd.concat([positive_candidates, remaining_positives], axis=0)

        # add positives to candidate_df
        positive_candidates["label_name"] = label
        positive_candidates["label_value"] = 1.0
        candidate_df = pd.concat([candidate_df, positive_candidates])

        # get negative candidates
        conditions = (df[label].isin([0.0, np.nan]))
        assert len(df[conditions]) >= args.num_negative, "subset we sample from must be large enough"
        negative_candidates = df[conditions].sample(n=args.num_negative, replace=False, random_state=seed)

        # add negatives to candidate_df
        negative_candidates["label_name"] = label
        negative_candidates["label_value"] = 0.0
        candidate_df = pd.concat([candidate_df, negative_candidates])

    return query_df, candidate_df


def split_train_val(df, seed):
    unique_subject_ids = df["subject_id"].unique()
    train_subject_ids, val_subject_ids = train_test_split(
        unique_subject_ids, test_size=0.10, shuffle=True, random_state=seed)

    train_df = df[df["subject_id"].isin(train_subject_ids)]
    val_df = df[df["subject_id"].isin(val_subject_ids)]

    return train_df, val_df


def run_assertions(train_df, val_df, test_query, test_candidate):
    assert set(train_df["subject_id"]).isdisjoint(set(val_df["subject_id"]))
    assert set(train_df["subject_id"]).isdisjoint(set(test_query["subject_id"]))
    assert set(train_df["subject_id"]).isdisjoint(set(test_candidate["subject_id"]))
    assert set(val_df["subject_id"]).isdisjoint(set(test_query["subject_id"]))
    assert set(val_df["subject_id"]).isdisjoint(set(test_candidate["subject_id"]))


if __name__ == '__main__':
    start = time.time()

    args = parse_create_splits()

    df = pd.read_csv(args.dataset_path)

    # get test set
    test_query, test_candidate = split_query_candidate(df, args, seed=args.seed)

    # split df so that there is no overlap in patient subject_id
    test_subject_ids = pd.concat([test_query["subject_id"], test_candidate["subject_id"]]).unique()
    train_val_df = df[~df["subject_id"].isin(test_subject_ids)]

    # get train and val sets
    train_df, val_df = split_train_val(train_val_df, seed=args.seed)

    # ensure there is no overlap in subject_id between train, val, and test sets
    run_assertions(train_df, val_df, test_query, test_candidate)

    # get val query and candidate sets
    val_query, val_candidate = split_query_candidate(val_df, args, seed=args.seed)

    # full test set is the candidate set without duplicates
    test_df = test_candidate.drop_duplicates(subset=["dicom_id"])

    # save splits
    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    test_df.to_csv(args.save_dir / "test.csv", index=False)

    val_query.to_csv(args.save_dir / "val_query.csv", index=False)
    val_candidate.to_csv(args.save_dir / "val_candidate.csv", index=False)
    test_query.to_csv(args.save_dir / "test_query.csv", index=False)
    test_candidate.to_csv(args.save_dir / "test_candidate.csv", index=False)

    with open(args.save_dir / "metadata.json", "w") as f:
        json.dump({
            "train set size": len(train_df),
            "val set size": len(val_df),
            "val query set size": len(val_query),
            "val candidate set size": len(val_candidate),
            "test set size": len(test_df),
            "test query set size": len(test_query),
            "test candidate set size": len(test_candidate)
        }, f, indent=4)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")