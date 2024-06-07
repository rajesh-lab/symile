import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.distributions.empirical_distribution import StepFunction

from args import parse_create_splits
from src.cxr_prediction.constants import LABS

# columns to drop from the dataset csv
COLS_TO_DROP = ['admittime', 'dischtime', 'deathtime', 'admission_type', 'admission_location',
                'discharge_location', 'race', 'hospital_expire_flag', 'gender', 'anchor_age',
                'anchor_year', 'dod', 'admittime_year', 'age', 'cxr_24_72_hr', 'cxr_dicom_id',
                'cxr_study_id', 'cxr_ViewPosition', 'cxr_ViewCodeSequence_CodeMeaning',
                'cxr_StudyDateTime', 'study_id','Enlarged Cardiomediastinum', 'Fracture',
                'Lung Lesion', 'Lung Opacity','Pleural Other', 'Pneumonia', 'Pneumothorax',
                'Support Devices', 'ecg_adm', 'ecg_study_id', 'ecg_file_name', 'ecg_time',
                'labs_all_nan']

LABEL_COLS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


class NaNAwareECDF(StepFunction):
    """
    Very similar to the statsmodels ECDF class
    (https://www.statsmodels.org/stable/generated/statsmodels.distributions.empirical_distribution.ECDF.html)
    except that it computes a NaN-aware ECDF by filling values corresponding to np.nan with np.nan.

    Source: https://stackoverflow.com/a/68959320
    """
    def __init__(self, x, side='right'):
        x = np.sort(x)

        # count number of non-nan's instead of length
        nobs = np.count_nonzero(~np.isnan(x))

        # fill the y values corresponding to np.nan with np.nan
        y = np.full_like(x, np.nan)
        y[:nobs]  = np.linspace(1./nobs,1,nobs)
        super(NaNAwareECDF, self).__init__(x, y, side=side, sorted=True)


def split_query_candidate(df, args, seed=42):
    """
    We focus on the five CheXpert labels.

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

    for label in LABEL_COLS:
        # get exclusive positive subset
        other_labels = [l for l in LABEL_COLS + ["No Finding"] if l != label]
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

    for label in LABEL_COLS:

        # get samples from query_df that are positive for this label
        positive_candidates = query_df[query_df.label_name == label]
        assert len(positive_candidates) == args.num_query

        # remove those samples from df to sample other positive samples
        num_remaining_positives = args.num_positive - len(positive_candidates)
        if num_remaining_positives > 0:
            df_filtered = df[~df["hadm_id"].isin(positive_candidates.hadm_id.to_list())]
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


def split_train_val(df, seed=42):
    unique_subject_ids = df["subject_id"].unique()
    train_subject_ids, val_subject_ids = train_test_split(
        unique_subject_ids, test_size=0.10, shuffle=True, random_state=seed)

    train_df = df[df["subject_id"].isin(train_subject_ids)]
    val_df = df[df["subject_id"].isin(val_subject_ids)]

    return train_df, val_df


def run_assertions(train_df, val_df, test_query, test_candidate):
    assert set(train_df["subject_id"]).isdisjoint(set(val_df["subject_id"]))
    assert set(train_df["hadm_id"]).isdisjoint(set(val_df["hadm_id"]))
    assert set(train_df["subject_id"]).isdisjoint(set(test_query["subject_id"]))
    assert set(train_df["subject_id"]).isdisjoint(set(test_candidate["subject_id"]))
    assert set(val_df["subject_id"]).isdisjoint(set(test_query["subject_id"]))
    assert set(val_df["subject_id"]).isdisjoint(set(test_candidate["subject_id"]))


def save_mean_percentiles(df, args):
    labs_means = {}
    for col in df.columns:
        if col.endswith("_percentile"):
            labs_means[col] = df[col].mean()

    with open(args.save_dir / "labs_means.json", "w") as f:
        json.dump(labs_means, f, indent=4)


if __name__ == '__main__':
    start = time.time()

    args = parse_create_splits()

    df = pd.read_csv(args.dataset_path).drop(columns=COLS_TO_DROP)

    # get test set
    test_query, test_candidate = split_query_candidate(df, args, seed=42)

    # split df so that there is no overlap in patient subject_id
    test_subject_ids = pd.concat([test_query["subject_id"], test_candidate["subject_id"]]).unique()
    train_val_df = df[~df["subject_id"].isin(test_subject_ids)]

    # get train and val sets
    train_df, val_df = split_train_val(train_val_df, seed=42)

    # ensure there is no overlap in subject_id between train, val, and test sets
    run_assertions(train_df, val_df, test_query, test_candidate)

    # normalize lab values and save mean percentiles
    for col in LABS.keys():
        ecdf = NaNAwareECDF(train_df[col])
        train_df[col + "_percentile"] = ecdf(train_df[col])
        val_df[col + "_percentile"] = ecdf(val_df[col])
        test_query[col + "_percentile"] = ecdf(test_query[col])
        test_candidate[col + "_percentile"] = ecdf(test_candidate[col])

    save_mean_percentiles(train_df, args)

    # get val query and candidate sets
    val_query, val_candidate = split_query_candidate(val_df, args, seed=42)

    # save splits
    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    val_query.to_csv(args.save_dir / "val_query.csv", index=False)
    val_candidate.to_csv(args.save_dir / "val_candidate.csv", index=False)
    test_query.to_csv(args.save_dir / "test_query.csv", index=False)
    test_candidate.to_csv(args.save_dir / "test_candidate.csv", index=False)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")