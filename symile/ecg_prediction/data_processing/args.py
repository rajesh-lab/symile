import argparse
from pathlib import Path

from symile.utils import str_to_bool


def parse_get_mimic_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mimiciv_hosp_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimiciv/2.2/hosp"),
                        help="Path to MIMIC-IV hospital module directory.")
    parser.add_argument("--ecg_data_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimic-iv-ecg/1.0"),
                        help="Directory with ECGs.")
    parser.add_argument("--save_dir", type=Path,
                        help="Where to save data.")

    return parser.parse_args()


def parse_create_splits():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/symile/cxr_prediction/datasets/dataset.csv"),
                        help="Path to dataframe with full dataset.")
    parser.add_argument("--save_dir", type=Path,
                        help="Where to save data.")

    parser.add_argument("--num_query", type=int, default=5,
                        help="Number of query samples per CXR label.")
    parser.add_argument("--num_positive", type=int, default=5,
                        help="Number of positive candidate samples per CXR label.")
    parser.add_argument("--num_negative", type=int, default=120,
                        help="Number of negative candidate samples per CXR label.")

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def parse_save_dataset_tensors():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split_to_run", type=str,
                        choices = ["train", "val_test"])

    ### DATASET ARGS ###
    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")
    parser.add_argument("--ecg_data_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimic-iv-ecg/1.0"),
                        help="Directory with ECGs.")
    parser.add_argument("--train_csv", type=Path,
                        default=Path("train.csv"),
                        help="Filename for train csv.")
    parser.add_argument("--val_csv", type=Path,
                        default=Path("val.csv"),
                        help="Filename for val csv.")
    parser.add_argument("--test_csv", type=Path,
                        default=Path("test.csv"),
                        help="Filename for val csv.")

    return parser.parse_args()