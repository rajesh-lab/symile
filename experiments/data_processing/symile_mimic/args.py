import argparse
from pathlib import Path


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg (str): String representing a boolean.

    Returns:
        Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_process_mimic_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mimiciv_hosp_dir", type=Path,
                        help="Path to MIMIC-IV hospital module directory, which \
                        must include the files patients.csv.gz, admissions.csv.gz, \
                            and labevents.csv.gz.")
    parser.add_argument("--cxr_data_dir", type=Path,
                        help="Directory with MIMIC CXR data, which must include the \
                        files mimic-cxr-2.0.0-metadata.csv.gz and mimic-cxr-2.0.0-chexpert.csv.gz.")
    parser.add_argument("--ecg_data_dir", type=Path,
                        help="Directory with MIMIC ECG data, which must include the \
                        file record_list.csv.")
    parser.add_argument("--save_dir", type=Path,
                        help="Where to save DataFrame with processed data.")

    return parser.parse_args()


def parse_create_dataset_splits():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=Path,
                        help="Path to csv file with full dataset.")
    parser.add_argument("--save_dir", type=Path,
                        help="Where to save data.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")

    parser.add_argument("--train_n", type=int,
                        help="Number of samples in train set.")
    parser.add_argument("--val_n", type=int,
                        help="Number of samples in val set.")
    parser.add_argument("--candidate_n", type=int,
                        help="Number of candidates for each test sample \
                              (candidate_n - 1 negative candidates for each).")

    return parser.parse_args()


def parse_process_and_save_tensors():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")
    parser.add_argument("--ecg_data_dir", type=Path,
                        help="Directory that contains the MIMIC `files` directory \
                              with ECG data.")
    parser.add_argument("--cxr_data_dir", type=Path,
                        help="Directory that contains the MIMIC `files` directory \
                              with CXR data.")
    parser.add_argument("--labs_means", type=Path,
                        default=Path("labs_means.json"),
                        help="json filename for labs means.")
    parser.add_argument("--train_csv", type=Path,
                        default=Path("train.csv"),
                        help="Filename for train csv.")
    parser.add_argument("--val_csv", type=Path,
                        default=Path("val.csv"),
                        help="Filename for val csv.")
    parser.add_argument("--val_retrieval_csv", type=Path,
                        default=Path("val_retrieval.csv"),
                        help="Filename for val retrieval csv.")
    parser.add_argument("--test_csv", type=Path,
                        default=Path("test.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--cxr_scale", type=int, default=320,
                        help="Scale for preprocessing CXRs.")
    parser.add_argument("--cxr_crop", type=int, default=320,
                        help="Crop for preprocessing CXRs.")

    return parser.parse_args()