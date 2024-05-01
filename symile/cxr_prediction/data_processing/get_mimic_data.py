import os
import time

import json

import numpy as np
import pandas as pd
import wfdb

from args import parse_get_mimic_data
from symile.cxr_prediction.constants import LABS


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

###########
### CXR ###
###########

def get_cxr_df_24_to_72(admissions_df, cxr_data_dir):
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

    # find the first CXR in the 24-72 hour period after admission
    def _find_cxr_24_72_hr(row):
        # filter for row subject's cxrs
        subject_df = cxr_df[cxr_df['subject_id'] == row['subject_id']]

        # get time difference between the admission time and each cxr's StudyDateTime
        time_diff = subject_df['StudyDateTime'] - row['admittime']

        # get CXRs within 24 to 72 hours after the admission time
        cxrs = subject_df[(time_diff > pd.Timedelta('24 hours')) & (time_diff <= pd.Timedelta('72 hours'))]

        # sort the CXRs by StudyDateTime to find the earliest one
        cxrs = cxrs.sort_values(by='StudyDateTime')

        # return the dicom_id of the first CXR in the sorted list if not empty
        return cxrs.iloc[0]['dicom_id'] if not cxrs.empty else None

    df = admissions_df.copy()
    df["cxr_24_72_hr"] = df.apply(_find_cxr_24_72_hr, axis=1)

    # only keep rows where cxr_24_72_hr is not None
    df = df[df["cxr_24_72_hr"].notna()]

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
                  left_on='cxr_24_72_hr', right_on='cxr_dicom_id', how='left')

    # add chexpert labels to df
    df = df.merge(chexpert_df.drop(columns='subject_id'), left_on='cxr_study_id', right_on='study_id', how='inner')

    return df


def get_cxr_df_post_24(admissions_df, cxr_data_dir):
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

    # find the first CXR after 24 hours after admission
    def _find_cxr_post_24_hr(row):
        # filter for row subject's cxrs
        subject_df = cxr_df[cxr_df['subject_id'] == row['subject_id']]

        # get time difference between the admission time and each cxr's StudyDateTime
        time_diff_adm = subject_df['StudyDateTime'] - row['admittime']

        # get time difference between each CXR's StudyDateTime and discharge time
        time_diff_disch = row['dischtime'] - subject_df['StudyDateTime']

        # get CXRs after 24 hours after admission and before discharge
        cxrs = subject_df[(time_diff_adm > pd.Timedelta('24 hours')) & (time_diff_disch > pd.Timedelta('0 days'))]

        # sort the CXRs by StudyDateTime to find the earliest one
        cxrs = cxrs.sort_values(by='StudyDateTime')

        # return the dicom_id of the first CXR in the sorted list if not empty
        return cxrs.iloc[0]['dicom_id'] if not cxrs.empty else None

    df = admissions_df.copy()
    df["cxr_post_24_hr"] = df.apply(_find_cxr_post_24_hr, axis=1)

    # only keep rows where cxr_post_24_hr is not None
    df = df[df["cxr_post_24_hr"].notna()]

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
                  left_on='cxr_post_24_hr', right_on='cxr_dicom_id', how='left')

    # add chexpert labels to df
    df = df.merge(chexpert_df.drop(columns='subject_id'), left_on='cxr_study_id', right_on='study_id', how='inner')

    return df

###########
### ECG ###
###########

def get_ecg_df_adm(admissions_df, ecg_data_dir):
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

def get_ecg_df_before_discharge(admissions_df, ecg_data_dir):
    ecg_df = pd.read_csv(f"{ecg_data_dir}/record_list.csv")

    ecg_df["ecg_time"] = pd.to_datetime(ecg_df["ecg_time"])
    ecg_df["full_path"] = ecg_df["path"].apply(lambda x: ecg_data_dir / x)

    # # remove ecg if signal is all zeros or if there are any nans
    # def _remove_ecg(pt):
    #     signal = wfdb.rdrecord(pt).p_signal
    #     return np.isnan(signal).any() or np.all(signal == 0)
    # remove_ecg_mask = ecg_df["full_path"].apply(_remove_ecg)
    # ecg_df = ecg_df[~remove_ecg_mask].drop("full_path", axis=1)

    # get ecgs from 24 hours before admission to 24 hours before discharge
    def _find_ecgs(row):
        # filter for row subject's ecgs
        subject_df = ecg_df[ecg_df['subject_id'] == row['subject_id']]

        # calculate time difference from admission and discharge
        adm_time_diff = subject_df['ecg_time'] - row['admittime']
        disc_time_diff = subject_df['ecg_time'] - row['dischtime']

        # get ecgs from 24 hours before admission to 24 hours before discharge
        ecgs = subject_df[(adm_time_diff >= pd.Timedelta('-24 hours')) & (disc_time_diff <= pd.Timedelta('-24 hours'))]

        # return a list of study_ids
        return ecgs['study_id'].tolist()

    df = admissions_df.copy()
    df["ecg_list"] = df.apply(_find_ecgs, axis=1)

    # expand ecg_list into separate rows for each hadm_id, ecg_study_id pair
    df = df.explode('ecg_list')
    df = df.rename(columns={'ecg_list': 'ecg_study_id'})

    # merge the admissions-based df with ecg_df to get the final ecg dataframe
    ecg_df = ecg_df.rename(columns={
        'study_id': 'ecg_study_id',
        'file_name': 'ecg_file_name',
        'ecg_time': 'ecg_time',
        'path': 'ecg_path'
    })

    df = df.merge(ecg_df[['ecg_study_id', 'ecg_file_name', 'ecg_time', 'ecg_path']],
                  on='ecg_study_id', how='left')

    df = df[df['ecg_study_id'].notna()]

    return df

############
### LABS ###
############

def get_labs_df_adm(admissions_df, mimiciv_hosp_dir):
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
        subject_df = labs_df[labs_df['subject_id'] == row['subject_id']]

        # get time difference between the admission time and each lab's time
        time_diff = subject_df['charttime'] - row['admittime']

        # get labs within 24 hours (before or after) the admission time
        labs = subject_df[(time_diff >= pd.Timedelta('-24 hours')) & (time_diff <= pd.Timedelta('24 hours'))]

        # return labevent_ids of labs
        return labs['labevent_id'].tolist() if not labs.empty else None

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


def get_labs_df_before_discharge(admissions_df, mimiciv_hosp_dir):
    labs_df = pd.read_csv(f"{mimiciv_hosp_dir}/labevents.csv.gz", compression='gzip')

    top_itemids = list(map(int, LABS.keys()))

    # filter to include only the top 50 labs
    labs_df = labs_df[labs_df["itemid"].isin(top_itemids)]
    labs_df["label"] = labs_df["itemid"].astype(str).map(LABS)

    labs_df["charttime"] = pd.to_datetime(labs_df["charttime"])

    # drop labs with missing values in `valuenum`
    labs_df = labs_df[~pd.isna(labs_df["valuenum"])]
    labs_df = labs_df[["labevent_id", "subject_id", "itemid", "charttime", "valuenum", "label"]]

    # get labs from 24 hours before admission to 24 hours before discharge
    def _find_labs(row):
        # filter for row subject's labs
        subject_df = labs_df[labs_df['subject_id'] == row['subject_id']]

        # calculate time difference from admission and discharge
        adm_time_diff = subject_df['charttime'] - row['admittime']
        disc_time_diff = subject_df['charttime'] - row['dischtime']

        # get labs from 24 hours before admission to 24 hours before discharge
        labs = subject_df[(adm_time_diff >= pd.Timedelta('-24 hours')) & (disc_time_diff <= pd.Timedelta('-24 hours'))]

        return labs['labevent_id'].tolist() if not labs.empty else None

    df = admissions_df.copy()
    df["labevent_ids"] = df.apply(_find_labs, axis=1)

    # only keep rows where labevent_ids is not None
    df = df[df["labevent_ids"].notna()]
    df = df[["subject_id", "hadm_id", "labevent_ids"]]

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

        for labevent_id in row["labevent_ids"]:
            if labevent_id in labs_df_dict:
                event_info = labs_df_dict[labevent_id]
                item_id = event_info["itemid"]
                if item_id in top_itemids:
                    # check if current labevent_id has an earlier charttime
                    if event_info["charttime"] < earliest_times[item_id]:
                        earliest_times[item_id] = event_info["charttime"]
                        # update df with valuenum of the earlier labevent_id
                        df.at[ix, item_id] = event_info["valuenum"]

    df = df.drop(columns=["labevent_ids"])
    df = df.dropna(subset=top_itemids, how="all")

    return df


if __name__ == '__main__':
    start = time.time()

    args = parse_get_mimic_data()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    admissions_df = get_admissions_df(args.mimiciv_hosp_dir)

    if args.data_type == "cxr":
        if args.cxr_type == "24_to_72":
            # cxrs within 24 to 72 hours of admission
            df = get_cxr_df_24_to_72(admissions_df, args.cxr_data_dir)
            df.to_csv(args.save_dir / "cxr_df.csv", index=False)
        elif args.cxr_type == "post_24":
            # cxrs after 24 hours of admission
            df = get_cxr_df_post_24(admissions_df, args.cxr_data_dir)
            df.to_csv(args.save_dir / "cxr_df.csv", index=False)
    elif args.data_type == "ecg":
        if args.ecg_labs_type == "adm":
            # ecgs within 24 hours of admission
            df = get_ecg_df_adm(admissions_df, args.ecg_data_dir)
            df.to_csv(args.save_dir / "ecg_df.csv", index=False)
        elif args.ecg_labs_type == "before_discharge":
            # ecgs anytime during admission
            df = get_ecg_df_before_discharge(admissions_df, args.ecg_data_dir)
            df.to_csv(args.save_dir / "ecg_df.csv", index=False)
    elif args.data_type == "labs":
        if args.ecg_labs_type == "adm":
            df = get_labs_df_adm(admissions_df, args.mimiciv_hosp_dir)
            df.to_csv(args.save_dir / "labs_df.csv", index=False)
        elif args.ecg_labs_type == "before_discharge":
            # labs anytime during admission
            df = get_labs_df_before_discharge(admissions_df, args.mimiciv_hosp_dir)
            df.to_csv(args.save_dir / "labs_df.csv", index=False)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")