import argparse
from pathlib import Path

from symile.utils import str_to_bool


def parse_get_mimic_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_type", type=str, default=None,
                        choices=["ecg", "labs"],
                        help="Type of data to extract.")

    parser.add_argument("--mimiciv_hosp_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimiciv/2.2/hosp"),
                        help="Path to MIMIC-IV hospital module directory.")
    parser.add_argument("--ecg_data_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimic-iv-ecg/1.0"),
                        help="Directory with ECGs.")
    parser.add_argument("--save_dir", type=Path,
                        help="Where to save data.")

    return parser.parse_args()


def parse_create_dataset():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ecg_df_path", type=Path,
                        help="Path to dataframe with ecg data.")
    parser.add_argument("--labs_df_path", type=Path,
                        help="Path to dataframe with labs data.")
    parser.add_argument("--save_dir", type=Path,
                        help="Where to save data.")

    return parser.parse_args()


def parse_save_dataset_tensors():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")
    parser.add_argument("--ecg_data_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimic-iv-ecg/1.0"),
                        help="Directory with ECGs.")
    parser.add_argument("--labs_means", type=Path,
                        default=Path("labs_means.json"),
                        help="json filename for labs means.")
    parser.add_argument("--train_csv", type=Path,
                        default=Path("train.csv"),
                        help="Filename for train csv.")
    parser.add_argument("--val_csv", type=Path,
                        default=Path("val.csv"),
                        help="Filename for val csv.")
    parser.add_argument("--test_csv", type=Path,
                        default=Path("test.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--split", type=str, default=None,
                        choices=["train", "val", "test"])

    return parser.parse_args()


def parse_create_mean_cxr_ecg():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")

    return parser.parse_args()