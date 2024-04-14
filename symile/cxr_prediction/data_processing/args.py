import argparse
from pathlib import Path

from symile.utils import str_to_bool


def parse_get_mimic_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_type", type=str, default=None,
                        choices=["cxr", "ecg", "labs"],
                        help="Type of data to extract.")

    parser.add_argument("--mimiciv_hosp_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/adriel/physionet.org/files/mimiciv/2.2/hosp"),
                        help="Path to MIMIC-IV hospital module directory.")
    parser.add_argument("--ecg_data_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimic-iv-ecg/1.0"),
                        help="Directory with ECGs.")
    parser.add_argument("--cxr_data_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimic-cxr-jpg/mimic-cxr-jpg-2.0.0.physionet.org"),
                        help="Directory with CXRs.")
    parser.add_argument("--save_dir", type=Path,
                        help="Where to save data.")

    return parser.parse_args()


def parse_save_dataset_tensors():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--data_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/healthcare/datasets"),
                        help="Directory with dataset csvs.")
    parser.add_argument("--ecg_data_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimic-iv-ecg/1.0"),
                        help="Directory with ECGs.")
    parser.add_argument("--cxr_data_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/mimic-cxr-jpg/mimic-cxr-jpg-2.0.0.physionet.org"),
                        help="Directory with CXRs.")
    parser.add_argument("--train_csv", type=Path,
                        default=Path("train.csv"),
                        help="Filename for train csv.")
    parser.add_argument("--val_csv", type=Path,
                        default=Path("val.csv"),
                        help="Filename for val csv.")
    parser.add_argument("--query_csv", type=Path,
                        default=Path("query.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--candidate_csv", type=Path,
                        default=Path("candidate.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--query_val_csv", type=Path,
                        default=Path("query_val.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--candidate_val_csv", type=Path,
                        default=Path("candidate_val.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--cxr_scale", type=int, default=320,
                        help="Scale for preprocessing CXRs.")
    parser.add_argument("--cxr_crop", type=int, default=320,
                        help="Crop for preprocessing CXRs.")

    return parser.parse_args()