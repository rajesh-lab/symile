"""
Processes MIMIC data to generate the Symile-MIMIC dataset.
"""
import os
import time

import numpy as np
import pandas as pd
import wfdb

from args import parse_process_mimic_data
from constants import LABS


def get_admissions_df(mimiciv_hosp_dir):
    """
    Retrieves and processes patient and admission data from the MIMIC-IV dataset.

    The `patients.csv` file contains patient information with unique identifiers `subject_id`.
    The `admissions.csv` file represents inpatient encounters with unique identifiers `hadm_id`.

    Args:
        mimiciv_hosp_dir (str): Directory path to the MIMIC-IV hospital data files.

    Returns:
        df (pd.DataFrame): contains merged and processed patient and admission data
            with a unique row for each `hadm_id`. The columns include: 'subject_id',
            'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type',
            'admission_location', 'discharge_location', 'race', 'hospital_expire_flag',
            'gender', 'anchor_age', 'anchor_year', 'dod', 'admittime_year', 'age'.
    """
    patients_df = pd.read_csv(f"{mimiciv_hosp_dir}/patients.csv.gz", compression="gzip")
    admissions_df = pd.read_csv(f"{mimiciv_hosp_dir}/admissions.csv.gz", compression="gzip")

    df = admissions_df.merge(patients_df, on="subject_id", how="left")

    df = df.drop(columns=["admit_provider_id", "insurance", "language", "marital_status",
                          "edregtime", "edouttime", "anchor_year_group"])

    # convert datetime columns to datetime type
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["dischtime"] = pd.to_datetime(df["dischtime"])
    df["deathtime"] = pd.to_datetime(df["deathtime"])

    # calculate patient's age in admission year
    df["admittime_year"] = df["admittime"].dt.year.astype("int64")
    df["age"] = df["anchor_age"] + (df["admittime_year"] - df["anchor_year"])

    return df

###########
### CXR ###
###########

def get_cxr_df(admissions_df, cxr_data_dir):
    """
    Retrieves and processes chest X-ray (CXR) data, merging it with admissions
    data from the MIMIC-IV dataset.

    Reads metadata and CheXpert labels from CXR files, filters and processes data,
    and for each admission identifies the earliest CXR taken 24 to 72 hours after
    admission.

    Args:
        admissions_df (pd.DataFrame): contains admissions information with unique identifier `hadm_id`.
        cxr_data_dir (str): Directory path to the CXR data files. Must include the files
                            `mimic-cxr-2.0.0-metadata.csv.gz` and `mimic-cxr-2.0.0-chexpert.csv.gz`.

    Returns:
        df (pd.DataFrame): Contains merged and processed CXR and admission data with a unique row for each `hadm_id`.
            Columns are: 'subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime',
            'admission_type', 'admission_location', 'discharge_location', 'race', 'hospital_expire_flag',
            'gender', 'anchor_age', 'anchor_year', 'dod', 'admittime_year', 'age', 'cxr_24_72_hr',
            'cxr_dicom_id', 'cxr_study_id', 'cxr_ViewPosition', 'cxr_ViewCodeSequence_CodeMeaning',
            'cxr_StudyDateTime', 'cxr_path', 'study_id', and columns for the CheXpert labels.
    """
    cxr_df = pd.read_csv(f"{cxr_data_dir}/mimic-cxr-2.0.0-metadata.csv.gz", compression="gzip")
    chexpert_df = pd.read_csv(f"{cxr_data_dir}/mimic-cxr-2.0.0-chexpert.csv.gz", compression="gzip")

    # get the StudyDateTime for each CXR
    cxr_df["StudyDate"] = cxr_df["StudyDate"].astype(str)
    cxr_df["StudyTime"] = cxr_df["StudyTime"].astype(int).astype(str).str.zfill(6)
    cxr_df["StudyDateTime"] = cxr_df["StudyDate"] + " " + cxr_df["StudyTime"]
    cxr_df["StudyDateTime"] = pd.to_datetime(cxr_df["StudyDateTime"], format="%Y%m%d %H%M%S")

    cxr_df = cxr_df.drop(columns=["StudyDate", "StudyTime", "ProcedureCodeSequence_CodeMeaning",
                                  "PatientOrientationCodeSequence_CodeMeaning"])

    # we only consider CXRs with a posteroanterior (PA) or anteroposterior (AP) view
    cxr_df = cxr_df[cxr_df["ViewPosition"].isin(["AP", "PA"])]

    # create a column with the path to the CXR image
    cxr_df["cxr_path"] = cxr_df.apply(
        lambda row: f"files/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg",
    axis=1)

    # filter out rows where the CXR image file does not exist
    file_exists_mask = cxr_df["cxr_path"].apply(lambda x: os.path.exists(os.path.join(cxr_data_dir, x)))
    cxr_df = cxr_df[file_exists_mask]

    # find the earliest CXR within 24 to 72 hours after admission
    def _find_cxr_24_72_hr(row):
        # filter for row subject's cxrs
        subject_df = cxr_df[cxr_df["subject_id"] == row["subject_id"]]

        # get time difference between the admission time and each cxr's StudyDateTime
        time_diff = subject_df["StudyDateTime"] - row["admittime"]

        # get CXRs within 24 to 72 hours after the admission time
        cxrs = subject_df[(time_diff > pd.Timedelta("24 hours")) & (time_diff <= pd.Timedelta("72 hours"))]

        # sort the CXRs by StudyDateTime to find the earliest one
        cxrs = cxrs.sort_values(by="StudyDateTime")

        # return the dicom_id of the first CXR in the sorted list if not empty
        return cxrs.iloc[0]["dicom_id"] if not cxrs.empty else None

    df = admissions_df.copy()
    df["cxr_24_72_hr"] = df.apply(_find_cxr_24_72_hr, axis=1)

    # only keep rows where cxr_24_72_hr is not None
    df = df[df["cxr_24_72_hr"].notna()]

    # merge the admissions-based df with cxr_df to get the final cxr dataframe
    cxr_df = cxr_df.rename(columns={
        "dicom_id": "cxr_dicom_id",
        "study_id": "cxr_study_id",
        "ViewPosition": "cxr_ViewPosition",
        "ViewCodeSequence_CodeMeaning": "cxr_ViewCodeSequence_CodeMeaning",
        "StudyDateTime": "cxr_StudyDateTime"
    })

    df = df.merge(cxr_df[["cxr_dicom_id", "cxr_study_id", "cxr_ViewPosition",
                          "cxr_ViewCodeSequence_CodeMeaning", "cxr_StudyDateTime", "cxr_path"]],
                  left_on="cxr_24_72_hr", right_on="cxr_dicom_id", how="left")

    # add chexpert labels to df
    df = df.merge(chexpert_df.drop(columns="subject_id"),
                  left_on="cxr_study_id", right_on="study_id", how="inner")

    return df

###########
### ECG ###
###########

def get_ecg_df(admissions_df, ecg_data_dir):
    """
    Retrieves and processes electrocardiogram (ECG) data, merging it with admissions
    data from the MIMIC-IV dataset.

    The function reads ECG records, filters and processes this data, and identifies
    the earliest ECG taken within 24 hours of admission.

    Args:
        admissions_df (pd.DataFrame): contains admissions information with unique identifier `hadm_id`.
        ecg_data_dir (str): Directory path to the ECG data files. Must include the file `record_list.csv`.

    Returns:
        df (pd.DataFrame): Contains merged and processed ECG and admission data with a unique row for each `hadm_id`.
            Columns are: 'subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime',
            'admission_type', 'admission_location', 'discharge_location', 'race', 'hospital_expire_flag',
            'gender', 'anchor_age', 'anchor_year', 'dod', 'admittime_year', 'age',
            'ecg_adm', 'ecg_study_id', 'ecg_file_name', 'ecg_time', 'ecg_path'.
    """
    ecg_df = pd.read_csv(f"{ecg_data_dir}/record_list.csv")

    ecg_df["ecg_time"] = pd.to_datetime(ecg_df["ecg_time"])
    ecg_df["full_path"] = ecg_df["path"].apply(lambda x: ecg_data_dir / x)

    # filter out rows where the ECG signal file is all zeros or contains any nans
    def _remove_ecg(pt):
        signal = wfdb.rdrecord(pt).p_signal
        return np.isnan(signal).any() or np.all(signal == 0)
    remove_ecg_mask = ecg_df["full_path"].apply(_remove_ecg)
    ecg_df = ecg_df[~remove_ecg_mask].drop("full_path", axis=1)

    # find the earliest ECG within 24 hours of admission
    def _find_ecg_adm(row):
        # filter for row subject's ecgs
        subject_df = ecg_df[ecg_df["subject_id"] == row["subject_id"]]

        # get time difference between the admission time and each ecg's time
        time_diff = subject_df["ecg_time"] - row["admittime"]

        # get ecgs within 24 hours (before or after) the admission time
        ecgs = subject_df[(time_diff >= pd.Timedelta("-24 hours")) & (time_diff <= pd.Timedelta("24 hours"))]

        # sort the ecgs by ecg_time to find the earliest one
        ecgs = ecgs.sort_values(by="ecg_time")

        # return the study_id of the first ECG in the sorted list if not empty
        return ecgs.iloc[0]["study_id"] if not ecgs.empty else None

    df = admissions_df.copy()
    df["ecg_adm"] = df.apply(_find_ecg_adm, axis=1)

    # only keep rows where ecg_adm is not None
    df = df[df["ecg_adm"].notna()]

    # merge the admissions-based df with ecg_df to get the final ecg dataframe
    ecg_df = ecg_df.rename(columns={
        "study_id": "ecg_study_id",
        "file_name": "ecg_file_name",
        "ecg_time": "ecg_time",
        "path": "ecg_path"
    })

    df = df.merge(ecg_df[["ecg_study_id", "ecg_file_name", "ecg_time", "ecg_path"]],
                  left_on="ecg_adm", right_on="ecg_study_id", how="left")

    return df

############
### LABS ###
############

def get_labs_df(admissions_df, mimiciv_hosp_dir):
    """
    Retrieves and processes laboratory events data, merging it with admissions
    data from the MIMIC-IV dataset.

    Reads laboratory events, filters for the top 50 labs, processes this data,
    and identifies the labs within 24 hours of admission.

    Args:
        admissions_df (pd.DataFrame): Containing admissions information with unique identifier `hadm_id`.
        mimiciv_hosp_dir (str): Directory path to the MIMIC-IV hospital data files. Must include the
                                file `labevents.csv.gz`.

    Returns:
        df (pd.DataFrame): Contains merged and processed laboratory events and admission data with a unique row
            for each `hadm_id`. The columns include: 'subject_id', 'hadm_id', and columns for each lab itemid
            representing the earliest lab value within 24 hours of admission.
    """
    labs_df = pd.read_csv(f"{mimiciv_hosp_dir}/labevents.csv.gz", compression='gzip')

    top_itemids = list(map(int, LABS.keys()))

    # filter to include only the top 50 labs
    labs_df = labs_df[labs_df["itemid"].isin(top_itemids)]
    labs_df["label"] = labs_df["itemid"].astype(str).map(LABS)

    labs_df["charttime"] = pd.to_datetime(labs_df["charttime"])

    # drop labs with missing values in `valuenum`
    labs_df = labs_df[~pd.isna(labs_df["valuenum"])]
    labs_df = labs_df[["labevent_id", "subject_id", "itemid", "charttime", "valuenum", "label"]]

    # find labs within 24 hours of admission
    def _find_labs_adm(row):
        # filter for row subject's labs
        subject_df = labs_df[labs_df["subject_id"] == row["subject_id"]]

        # get time difference between the admission time and each lab's time
        time_diff = subject_df["charttime"] - row["admittime"]

        # get labs within 24 hours (before or after) the admission time
        labs = subject_df[(time_diff >= pd.Timedelta("-24 hours")) & (time_diff <= pd.Timedelta("24 hours"))]

        # return labevent_ids of labs
        return labs["labevent_id"].tolist() if not labs.empty else None

    df = admissions_df.copy()
    df["labs_adm"] = df.apply(_find_labs_adm, axis=1)

    # only keep rows where labs_adm is not None
    df = df[df["labs_adm"].notna()]
    df = df[["subject_id", "hadm_id", "labs_adm"]]

    # initialize columns for each lab itemid and then find the earliest lab value
    # for each itemid within 24 hours of that admission
    for item_id in top_itemids:
        df[item_id] = np.nan

    # create dictionary from labs_df for quick lookups
    labs_df_dict = labs_df.set_index("labevent_id")[["itemid", "valuenum", "charttime"]].to_dict("index")

    # initialize a placeholder for the earliest charttime per itemid for each row in df
    earliest_times_temp = {item_id: pd.Timestamp.max for item_id in top_itemids}

    for ix, row in df.iterrows():
        earliest_times = earliest_times_temp.copy()

        for labevent_id in row["labs_adm"]:
            if labevent_id in labs_df_dict:
                event_info = labs_df_dict[labevent_id]
                item_id = event_info["itemid"]
                if item_id in top_itemids:
                    # check if current labevent_id has an earlier charttime
                    if event_info["charttime"] < earliest_times[item_id]:
                        earliest_times[item_id] = event_info["charttime"]
                        # update df with valuenum of the earlier labevent_id
                        df.at[ix, item_id] = event_info["valuenum"]

    df = df.drop(columns=["labs_adm"])
    df = df.dropna(subset=top_itemids, how="all")

    return df


def merge_dataframes(cxr_df, ecg_df, labs_df):
    """
    Merges the CXR, ECG, and laboratory dataframes, ensuring that each row contains
    data from all three sources for each admission.
    """
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

    args = parse_process_mimic_data()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # get admissions data
    print("Processing admissions data...")
    admissions_df = get_admissions_df(args.mimiciv_hosp_dir)

    # get cxrs within 24 to 72 hours of admission
    print("Processing CXR data...")
    cxr_df = get_cxr_df(admissions_df, args.cxr_data_dir)

    # get ecgs within 24 hours of admission
    print("Processing ECG data...")
    ecg_df = get_ecg_df(admissions_df, args.ecg_data_dir)

    # get labs within 24 hours of admission
    print("Processing labs data...")
    labs_df = get_labs_df(admissions_df, args.mimiciv_hosp_dir)

    print("Merging dataframes...")
    df = merge_dataframes(cxr_df, ecg_df, labs_df)

    df.to_csv(args.save_dir / "symile_mimic_data.csv", index=False)

    print("length of dataset: ", len(df))
    print("number of unique admissions (should be same as length of dataset): ", df["hadm_id"].nunique())
    print("number of unique subjects: ", df["subject_id"].nunique())

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")