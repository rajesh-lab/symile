import json

import pandas as pd
from sklearn.model_selection import train_test_split

from args import parse_create_dataset

ADM_COLS = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
            "discharge_location", "hospital_expire_flag", "dod"]
CXR_COLS = ["cxr_dicom_id", "cxr_study_id", "cxr_StudyDateTime", "cxr_path"]
ECG_COLS = ["ecg_study_id", "ecg_time", "ecg_path"]


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
    inhospital_deaths_df = df[df.hospital_expire_flag == 1]

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
    cxr_df = pd.read_csv(args.cxr_df_path)[ADM_COLS + CXR_COLS]
    ecg_df = pd.read_csv(args.ecg_df_path)[ADM_COLS + ECG_COLS]

    # so that each row has both cxr and ecg
    df = pd.merge(cxr_df, ecg_df, on=ADM_COLS, how="inner")

    for col in ["admittime", "dischtime", "deathtime", "dod", "cxr_StudyDateTime", "ecg_time"]:
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

    train_df.to_csv(args.save_dir / "train.csv", index=False)
    val_df.to_csv(args.save_dir / "val.csv", index=False)
    test_df.to_csv(args.save_dir / "test.csv", index=False)

    with open(args.save_dir / "dataset_sizes.json", 'w') as f:
        json.dump({
            "train set size": len(train_df),
            "val set size": len(val_df),
            "test set size": len(test_df)
        }, f, indent=4)