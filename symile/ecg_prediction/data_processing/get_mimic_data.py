import os
import time

import numpy as np
import pandas as pd
import wfdb

from args import parse_get_mimic_data


def get_patients_df(mimiciv_hosp_dir):
    """
    The main patients table is `patients.csv` with unique identifer `subject_id`.
    """
    df = pd.read_csv(f"{mimiciv_hosp_dir}/patients.csv.gz", compression='gzip')
    df = df.drop(columns=['anchor_year_group', 'dod'])
    return df


def get_ecg_df(ecg_data_dir):
    ecg_df = pd.read_csv(f"{ecg_data_dir}/record_list.csv")

    ecg_df["ecg_time"] = pd.to_datetime(ecg_df["ecg_time"])
    ecg_df["full_path"] = ecg_df["path"].apply(lambda x: ecg_data_dir / x)

    # remove ecg if signal is all zeros or if there are any nans
    def _remove_ecg(pt):
        signal = wfdb.rdrecord(pt).p_signal
        return np.isnan(signal).any() or np.all(signal == 0)
    remove_ecg_mask = ecg_df["full_path"].apply(_remove_ecg)
    ecg_df = ecg_df[~remove_ecg_mask].drop("full_path", axis=1)

    return ecg_df


def merge_patients_ecgs(patients_df, ecg_df):
    df = ecg_df.merge(patients_df, on='subject_id', how='left')

    # calculate patient's age in admission year
    df['ecg_year'] = df['ecg_time'].dt.year
    df['age'] = df['anchor_age'] + (df['ecg_year'] - df['anchor_year'])
    df = df.drop(columns=['ecg_year'])

    df = df.dropna(subset=['age', 'gender'])

    return df


if __name__ == '__main__':
    start = time.time()

    args = parse_get_mimic_data()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    patients_df = get_patients_df(args.mimiciv_hosp_dir)

    ecg_df = get_ecg_df(args.ecg_data_dir)

    df = merge_patients_ecgs(patients_df, ecg_df)

    df.to_csv(args.save_dir / "df.csv", index=False)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")