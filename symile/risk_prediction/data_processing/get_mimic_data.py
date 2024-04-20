import os
import time

import numpy as np
import pandas as pd
import wfdb

from args import parse_get_mimic_data


def get_admissions_df(mimiciv_hosp_dir):
    """
    The main patients table is `patients.csv` with unique identifer `subject_id`.

    The table `admissions.csv` with unique identifer `hadm_id` represents a
    single inpatient encounter. Any table without `hadm_id` pertains to data
    collected outside of an inpatient encounter.
    """
    patients_df = pd.read_csv(f"{mimiciv_hosp_dir}/patients.csv.gz", compression='gzip')
    admissions_df = pd.read_csv(f"{mimiciv_hosp_dir}/admissions.csv.gz", compression='gzip')

    df = admissions_df.merge(patients_df, on='subject_id', how='left')

    df = df.drop(columns=['admit_provider_id', 'insurance', 'language', 'marital_status', 'edregtime', 'edouttime', 'anchor_year_group'])

    # convert datetime columns to datetime type
    df['admittime'] = pd.to_datetime(df['admittime'])
    df['dischtime'] = pd.to_datetime(df['dischtime'])
    df['deathtime'] = pd.to_datetime(df['deathtime'])

    # calculate patient's age in admission year
    df['admittime_year'] = df['admittime'].dt.year.astype('int64')
    df['age'] = df['anchor_age'] + (df['admittime_year'] - df['anchor_year'])

    return df


def get_cxr_df(admissions_df, cxr_data_dir):
    cxr_df = pd.read_csv(f"{cxr_data_dir}/mimic-cxr-2.0.0-metadata.csv.gz", compression='gzip')
    chexpert_df = pd.read_csv(f"{cxr_data_dir}/mimic-cxr-2.0.0-chexpert.csv.gz", compression='gzip')

    cxr_df["StudyDate"] = cxr_df["StudyDate"].astype(str)
    cxr_df["StudyTime"] = cxr_df["StudyTime"].astype(int).astype(str).str.zfill(6)
    cxr_df["StudyDateTime"] = cxr_df["StudyDate"] + " " + cxr_df["StudyTime"]
    cxr_df["StudyDateTime"] = pd.to_datetime(cxr_df["StudyDateTime"], format="%Y%m%d %H%M%S")

    cxr_df = cxr_df.drop(columns=["StudyDate", "StudyTime", "ProcedureCodeSequence_CodeMeaning", "PatientOrientationCodeSequence_CodeMeaning"])

    # we only consider CXRs with a posteroanterior (PA) or anteroposterior (AP) view
    cxr_df = cxr_df[cxr_df['ViewPosition'].isin(['AP', 'PA'])]

    cxr_df['cxr_path'] = cxr_df.apply(
        lambda row: f"files/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg",
    axis=1)

    # only keep rows where the CXR exists
    file_exists_mask = cxr_df["cxr_path"].apply(lambda x: os.path.exists(os.path.join(cxr_data_dir, x)))
    cxr_df = cxr_df[file_exists_mask]

    # find CXR within 24 hours of admission
    def _find_cxr_adm(row):
        # filter for row subject's cxrs
        subject_df = cxr_df[cxr_df['subject_id'] == row['subject_id']]

        # get time difference between the admission time and each cxr's StudyDateTime
        time_diff = subject_df['StudyDateTime'] - row['admittime']

        # get CXRs within 24 hours (before or after) of the admission time
        cxrs = subject_df[(time_diff >= pd.Timedelta('-24 hours')) & (time_diff <= pd.Timedelta('24 hours'))]

        # sort the CXRs by StudyDateTime to find the earliest one
        cxrs = cxrs.sort_values(by='StudyDateTime')

        # return the dicom_id of the first CXR in the sorted list if not empty
        return cxrs.iloc[0]['dicom_id'] if not cxrs.empty else None

    df = admissions_df.copy()
    df["cxr_adm"] = df.apply(_find_cxr_adm, axis=1)

    # only keep rows where cxr_adm is not None
    df = df[df["cxr_adm"].notna()]

    # merge the admissions-based df with cxr_df to get the final cxr dataframe
    cxr_df = cxr_df.rename(columns={
        'dicom_id': 'cxr_dicom_id',
        'study_id': 'cxr_study_id',
        'ViewPosition': 'cxr_ViewPosition',
        'ViewCodeSequence_CodeMeaning': 'cxr_ViewCodeSequence_CodeMeaning',
        'StudyDateTime': 'cxr_StudyDateTime'
    })

    df = df.merge(cxr_df[['cxr_dicom_id', 'cxr_study_id', 'cxr_ViewPosition',
                          'cxr_ViewCodeSequence_CodeMeaning', 'cxr_StudyDateTime', 'cxr_path']],
                  left_on='cxr_adm', right_on='cxr_dicom_id', how='left')

    # add chexpert labels to df
    df = df.merge(chexpert_df.drop(columns='subject_id'), left_on='cxr_study_id', right_on='study_id', how='inner')

    return df


def get_ecg_df(admissions_df, ecg_data_dir):
    ecg_df = pd.read_csv(f"{ecg_data_dir}/record_list.csv")

    ecg_df["ecg_time"] = pd.to_datetime(ecg_df["ecg_time"])
    ecg_df["full_path"] = ecg_df["path"].apply(lambda x: ecg_data_dir / x)

    # remove ecg if signal is all zeros or if there are any nans
    def _remove_ecg(pt):
        signal = wfdb.rdrecord(pt).p_signal
        return np.isnan(signal).any() or np.all(signal == 0)
    remove_ecg_mask = ecg_df["full_path"].apply(_remove_ecg)
    ecg_df = ecg_df[~remove_ecg_mask].drop("full_path", axis=1)

    # find earliest ECG within 24 hours of admission
    def _find_ecg_adm(row):
        # filter for row subject's ecgs
        subject_df = ecg_df[ecg_df['subject_id'] == row['subject_id']]

        # get time difference between the admission time and each ecg's time
        time_diff = subject_df['ecg_time'] - row['admittime']

        # get ecgs within 24 hours (before or after) the admission time
        ecgs = subject_df[(time_diff >= pd.Timedelta('-24 hours')) & (time_diff <= pd.Timedelta('24 hours'))]

        # sort the ecgs by ecg_time to find the earliest one
        ecgs = ecgs.sort_values(by='ecg_time')

        # return the study_id of the first ECG in the sorted list if not empty
        return ecgs.iloc[0]['study_id'] if not ecgs.empty else None

    df = admissions_df.copy()
    df["ecg_adm"] = df.apply(_find_ecg_adm, axis=1)

    # only keep rows where ecg_adm is not None
    df = df[df["ecg_adm"].notna()]

    # merge the admissions-based df with ecg_df to get the final ecg dataframe
    ecg_df = ecg_df.rename(columns={
        'study_id': 'ecg_study_id',
        'file_name': 'ecg_file_name',
        'ecg_time': 'ecg_time',
        'path': 'ecg_path'
    })

    df = df.merge(ecg_df[['ecg_study_id', 'ecg_file_name', 'ecg_time', 'ecg_path']],
                  left_on='ecg_adm', right_on='ecg_study_id', how='left')

    return df


if __name__ == '__main__':
    start = time.time()

    args = parse_get_mimic_data()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    admissions_df = get_admissions_df(args.mimiciv_hosp_dir)

    if args.data_type == "cxr":
        df = get_cxr_df(admissions_df, args.cxr_data_dir)
        df.to_csv(args.save_dir / "cxr_df.csv", index=False)
    elif args.data_type == "ecg":
        df = get_ecg_df(admissions_df, args.ecg_data_dir)
        df.to_csv(args.save_dir / "ecg_df.csv", index=False)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")