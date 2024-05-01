import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.distributions.empirical_distribution import StepFunction

from args import parse_create_dataset
from symile.risk_prediction.constants import LABS

ADM_COLS = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
            "discharge_location", "hospital_expire_flag", "dod"]
ECG_COLS = ["ecg_study_id", "ecg_time", "ecg_path"]


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


def length_of_stay(df):
    """
    `admittime` provides the date and time the patient was admitted to the hospital,
    while `dischtime` provides the date and time the patient was discharged from the hospital.

    We first create a column called `length_of_stay_days`. An LOS of 0 days indicates a stay
    that's less than 24 hours. An LOS of 1 day indicates a stay that's between 24 and 48 hours.

    According to the MIMIC documentation, "Organ donor accounts are sometimes created for
    patients who died in the hospital. These are distinct hospital admissions with very short,
    sometimes negative lengths of stay. Furthermore, their `deathtime` is frequently the same as
    the earlier patient admission’s `deathtime`." Therefore, we'll only keep rows where LOS >= 0.

    Note that we include patients who died in-hospital, whose `dischtime` is the same as `deathtime`.
    """
    df["length_of_stay_days"] = (df["dischtime"] - df["admittime"]).dt.days
    df = df[df["length_of_stay_days"] >= 0]

    return df


def mortality(df):
    """
    If applicable, `deathtime` provides the time of in-hospital death for the patient. Note that
    `deathtime` is only present if the patient died in-hospital, and is almost always the same as
    the patient’s `dischtime`. However, there can be some discrepancies due to typographical errors.

    `deathtime` is the time of death of a patient if they died in hospital. If the patient did not
    die within the hospital for the given hospital admission, `deathtime` will be null.

    Note that people discharged into hospice are not considered an in-hospital death.

    `hospital_expire_flag` is a binary flag which indicates whether the patient died within the given
    hospitalization. 1 indicates death in the hospital, and 0 indicates survival to hospital discharge.

    Whenever there are inconsistencies between `deathtime` and `hospital_expire_flag`,
    `hospital_expire_flag` is the correct column (because `discharge_location` is consistent with `dod`).
    So we can use `hospital_expire_flag` as the label.
    """
    # We'll remove all rows where `dod` is before `dischtime` since these are likely organ donor accounts.
    df = df[~(df["dod"].dt.date < df["dischtime"].dt.date)]

    return df


def readmission(df):
    # takes about 8 minutes
    def _find_admission_within_30_days(row):
        # filter for row subject's admissions
        subject_df = df[df["subject_id"] == row["subject_id"]]

        # filter for admissions after the current discharge time
        future_admissions = subject_df[subject_df["admittime"] > row["dischtime"]]

        # find admissions within 30 days, if any
        within_30_days = future_admissions[future_admissions["admittime"] <= row["dischtime"] + pd.Timedelta(days=30)]

        if within_30_days.empty:
            return 0
        else:
            return 1

    df["adm_within_30_days"] = df.apply(_find_admission_within_30_days, axis=1)

    return df


def create_dataset(args):
    ecg_df = pd.read_csv(args.ecg_df_path)[ADM_COLS + ECG_COLS]
    labs_df = pd.read_csv(args.labs_df_path)

    # add a column to labs_df in order to confirm that not all values are nan
    labs_cols = labs_df.columns.drop(["subject_id", "hadm_id"])
    labs_df["labs_all_nan"] = labs_df[labs_cols].isna().all(axis=1).astype(int)

    # merge dataframes
    df = pd.merge(ecg_df, labs_df, on=["subject_id", "hadm_id"], how="inner")

    # make sure each row has ecg and at least one lab
    assert df["ecg_study_id"].notna().all(), "'ecg_study_id' contains NaN values."
    assert df["labs_all_nan"].eq(0).all(), "Each row should have at least one lab value."

    for col in ["admittime", "dischtime", "deathtime", "dod", "ecg_time"]:
        df[col] = pd.to_datetime(df[col])

    df = length_of_stay(df)
    df = mortality(df)
    df = readmission(df)

    # get splits ensuring that there is no patient overlap
    unique_subject_ids = df["subject_id"].unique()
    train_subject_ids, val_subject_ids = train_test_split(unique_subject_ids, train_size=0.8, shuffle=True, random_state=42)
    val_subject_ids, test_subject_ids = train_test_split(val_subject_ids, test_size=0.5, shuffle=True, random_state=42)

    train_df = df[df["subject_id"].isin(train_subject_ids)]
    val_df = df[df["subject_id"].isin(val_subject_ids)]
    test_df = df[df["subject_id"].isin(test_subject_ids)]

    assert set(train_df["subject_id"]).isdisjoint(set(val_df["subject_id"]))
    assert set(train_df["subject_id"]).isdisjoint(set(test_df["subject_id"]))
    assert set(val_df["subject_id"]).isdisjoint(set(test_df["subject_id"]))

    # create length of stay quantiles using the train set, and then apply them to the val and test sets.
    train_df["los_quantile"], train_bins = pd.qcut(
        train_df["length_of_stay_days"], q=4, labels=False, retbins=True, duplicates="raise"
    )

    val_df["los_quantile"], val_bins = pd.cut(
        val_df["length_of_stay_days"], bins=train_bins, labels=False, retbins=True, include_lowest=True
    )

    test_df["los_quantile"], test_bins = pd.cut(
        test_df["length_of_stay_days"], bins=train_bins, labels=False, retbins=True, include_lowest=True
    )

    return (train_df, val_df, test_df)


if __name__ == '__main__':
    args = parse_create_dataset()

    (train_df, val_df, test_df) = create_dataset(args)

    # normalize lab values and save mean percentiles
    for col in LABS.keys():
        ecdf = NaNAwareECDF(train_df[col])

        train_df[col + "_percentile"] = ecdf(train_df[col])
        val_df[col + "_percentile"] = ecdf(val_df[col])
        test_df[col + "_percentile"] = ecdf(test_df[col])

    save_mean_percentiles(train_df, args)

    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    test_df.to_csv(args.save_dir / "test.csv", index=False)

    with open(args.save_dir / "dataset_sizes.json", 'w') as f:
        json.dump({
            "train set size": len(train_df),
            "val set size": len(val_df),
            "test set size": len(test_df)
        }, f, indent=4)