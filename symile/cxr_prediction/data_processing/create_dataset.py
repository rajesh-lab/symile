import time

import pandas as pd

from args import parse_create_dataset


def create_dataset(args):
    cxr_df = pd.read_csv(args.cxr_df_path)
    ecg_df = pd.read_csv(args.ecg_df_path)
    labs_df = pd.read_csv(args.labs_df_path)

    # add a column to labs_df in order to confirm that not all values are nan
    labs_cols = labs_df.columns.drop(["subject_id", "hadm_id"])
    labs_df["labs_all_nan"] = labs_df[labs_cols].isna().all(axis=1).astype(int)

    # drop admissions-related columns from ecg_df to allow merging
    # (cxr_df will keep admissions-related columns)
    ecg_cols_to_drop = ["admittime", "dischtime", "deathtime", "admission_type",
        "admission_location", "discharge_location", "race", "hospital_expire_flag",
        "gender", "anchor_age", "anchor_year", "dod", "admittime_year", "age"]
    ecg_df = ecg_df.drop(columns=ecg_cols_to_drop)

    # merge dataframes
    df = pd.merge(cxr_df, ecg_df, on=["subject_id", "hadm_id"], how="inner")
    df = pd.merge(df, labs_df, on=["subject_id", "hadm_id"], how="inner")

    # make sure there are no rows missing cxr, ecg, or at least one lab
    assert df["cxr_dicom_id"].notna().all(), "'cxr_dicom_id' contains NaN values."
    assert df["ecg_study_id"].notna().all(), "'ecg_study_id' contains NaN values."
    assert df["labs_all_nan"].eq(0).all(), "Each row should have at least one lab value."

    return df


if __name__ == '__main__':
    start = time.time()

    args = parse_create_dataset()

    df = create_dataset(args)

    df.to_csv(args.save_dir / "dataset.csv", index=False)

    print("length of dataset: ", len(df))
    print("number of unique admissions (should be same as length of dataset): ", df["hadm_id"].nunique())
    print("number of unique subjects: ", df["subject_id"].nunique())

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")