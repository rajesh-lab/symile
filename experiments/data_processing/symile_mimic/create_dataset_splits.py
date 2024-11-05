import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.distributions.empirical_distribution import StepFunction

from args import parse_create_dataset_splits
from constants import LABS

# admissions-related columns to drop from the dataset csv
COLS_TO_DROP = ["admittime", "dischtime", "deathtime", "admission_type", "admission_location",
                "discharge_location", "race", "hospital_expire_flag", "gender", "anchor_age",
                "anchor_year", "dod", "admittime_year", "age", "cxr_24_72_hr", "cxr_dicom_id",
                "cxr_study_id", "cxr_ViewPosition", "cxr_ViewCodeSequence_CodeMeaning",
                "cxr_StudyDateTime", "study_id", "Consolidation", "Enlarged Cardiomediastinum",
                "Fracture", "Lung Lesion", "Pleural Other", "Pneumonia", "Pneumothorax",
                "Support Devices", "ecg_adm", "ecg_study_id", "ecg_file_name", "ecg_time",
                "labs_all_nan"]


def run_assertions(train_df, val_df, test_df):
    assert set(train_df["subject_id"]).isdisjoint(set(val_df["subject_id"]))
    assert set(train_df["hadm_id"]).isdisjoint(set(val_df["hadm_id"]))
    assert set(train_df["subject_id"]).isdisjoint(set(test_df["subject_id"]))
    assert set(train_df["hadm_id"]).isdisjoint(set(test_df["hadm_id"]))
    assert set(val_df["subject_id"]).isdisjoint(set(test_df["subject_id"]))
    assert set(val_df["hadm_id"]).isdisjoint(set(test_df["hadm_id"]))


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


def save_mean_percentiles(df, args):
    labs_means = {}
    for col in df.columns:
        if col.endswith("_percentile"):
            labs_means[col] = df[col].mean()

    with open(args.save_dir / "labs_means.json", "w") as f:
        json.dump(labs_means, f, indent=4)


def sample_negative_candidates(df, candidate_n):
    """
    Generates negative samples for each row in df. The resulting DataFrame will
    contain positive samples (original rows) labeled as 1, and negative samples
    (randomly sampled rows) labeled as 0. Each negative sample is assigned the same
    'label_hadm_id' as its corresponding positive sample to indicate the
    hadm_id it is a negative candidate for.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        candidate_n (int): The number of candidates for each test sample
                           (candidate_n - 1 negative candidates to sample).

    Returns:
        pd.DataFrame: The DataFrame containing both the original rows with label 1
                      and the negative samples with label 0.
    """
    # Each sample is a "query" sample for that hadm_id, so we'll label them as "positive".
    df["label_hadm_id"] = df["hadm_id"]
    df["label"] = 1

    sampled_rows = []

    for i in range(len(df)):
        # sample candidate_n - 1 rows to be negative candidates for the current row
        remaining_df = df.drop(df.index[i])
        neg_candidates = remaining_df.sample(n=candidate_n-1, replace=False)

        neg_candidates["label_hadm_id"] = df.iloc[i]["hadm_id"]
        neg_candidates["label"] = 0

        sampled_rows.append(neg_candidates)

    return pd.concat([df] + sampled_rows, ignore_index=True)


if __name__ == '__main__':
    start = time.time()

    args = parse_create_dataset_splits()

    seed = args.seed if args.use_seed else None

    df = pd.read_csv(args.dataset_path).drop(columns=COLS_TO_DROP)

    train_df, val_test_df = train_test_split(df, train_size=args.train_n,
                                             shuffle=True, random_state=seed)

    # make sure there's no overlap in subject_id between train and val/test sets
    train_subject_ids = train_df["subject_id"].unique()
    val_test_df = val_test_df[~val_test_df["subject_id"].isin(train_subject_ids)]

    val_df, test_df = train_test_split(val_test_df, train_size=args.val_n,
                                       shuffle=True, random_state=seed)

    # make sure there's no overlap in subject_id between val and test sets
    val_subject_ids = val_df["subject_id"].unique()
    test_df = test_df[~test_df["subject_id"].isin(val_subject_ids)]

    # ensure (again) that there is no overlap in subject_id across all three splits
    run_assertions(train_df, val_df, test_df)

    # normalize lab values and save mean percentiles
    for col in LABS.keys():
        ecdf = NaNAwareECDF(train_df[col])
        train_df[col + "_percentile"] = ecdf(train_df[col])
        val_df[col + "_percentile"] = ecdf(val_df[col])
        test_df[col + "_percentile"] = ecdf(test_df[col])

    save_mean_percentiles(train_df, args)

    # sample negative candidates for evaluation sets
    val_retrieval_df = sample_negative_candidates(val_df, args.candidate_n)
    test_df = sample_negative_candidates(test_df, args.candidate_n)

    # save splits
    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    val_retrieval_df.to_csv(args.save_dir / "val_retrieval.csv", index=False)
    test_df.to_csv(args.save_dir / "test.csv", index=False)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")