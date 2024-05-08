import os
import time

import numpy as np
import pandas as pd

from args import parse_get_mimic_data


def get_patients_df(mimiciv_hosp_dir):
    """
    The main patients table is `patients.csv` with unique identifer `subject_id`.
    """
    df = pd.read_csv(f"{mimiciv_hosp_dir}/patients.csv.gz", compression='gzip')
    df = df.drop(columns=['anchor_year_group', 'dod'])
    return df


def get_cxr_df(cxr_data_dir, pa_ap_only):
    chexpert_df = pd.read_csv(f"{cxr_data_dir}/mimic-cxr-2.0.0-chexpert.csv.gz", compression='gzip')

    df = pd.read_csv(f"{cxr_data_dir}/mimic-cxr-2.0.0-metadata.csv.gz", compression='gzip')

    df["StudyDate"] = df["StudyDate"].astype(str)
    df["StudyTime"] = df["StudyTime"].astype(int).astype(str).str.zfill(6)
    df["StudyDateTime"] = df["StudyDate"] + " " + df["StudyTime"]
    df["StudyDateTime"] = pd.to_datetime(df["StudyDateTime"], format="%Y%m%d %H%M%S")

    df = df.drop(columns=["StudyDate", "StudyTime", "ProcedureCodeSequence_CodeMeaning",
                          "PatientOrientationCodeSequence_CodeMeaning", "Rows", "Columns",
                          "PerformedProcedureStepDescription"])

    if pa_ap_only:
        # only consider CXRs with a posteroanterior (PA) or anteroposterior (AP) view
        df = df[df['ViewPosition'].isin(['AP', 'PA'])]

    df['cxr_path'] = df.apply(
        lambda row: f"files/p{str(row['subject_id'])[:2]}/p{row['subject_id']}/s{row['study_id']}/{row['dicom_id']}.jpg",
        axis=1
    )

    # only keep rows where the CXR file exists
    file_exists_mask = df["cxr_path"].apply(lambda x: os.path.exists(os.path.join(cxr_data_dir, x)))
    df = df[file_exists_mask]

    # add chexpert labels to df
    df = df.merge(chexpert_df, on=['subject_id', 'study_id'], how='left')

    return df


def merge_patients_cxrs(patients_df, cxr_df):
    df = cxr_df.merge(patients_df, on='subject_id', how='left')

    # calculate patient's age in admission year
    df['cxr_year'] = df['StudyDateTime'].dt.year
    df['age'] = df['anchor_age'] + (df['cxr_year'] - df['anchor_year'])
    df = df.drop(columns=['cxr_year'])

    df = df.dropna(subset=['age', 'gender'])

    return df


if __name__ == '__main__':
    start = time.time()

    args = parse_get_mimic_data()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    patients_df = get_patients_df(args.mimiciv_hosp_dir)

    cxr_df = get_cxr_df(args.cxr_data_dir, args.pa_ap_only)

    df = merge_patients_cxrs(patients_df, cxr_df)

    filename = "df_pa_ap_only.csv" if args.pa_ap_only else "df.csv"
    df.to_csv(args.save_dir / filename, index=False)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")