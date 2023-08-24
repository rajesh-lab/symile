import argparse
from pathlib import Path

from utils import str_to_bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_language", type=int, default=1,
                        help="Number of samples per language in pretraining dataset.")
    parser.add_argument("--audio_save_dir", type=Path,
                        default=Path("./audio"),
                        help="Where to save generated audio files.")
    parser.add_argument("--commonvoice_dir", type=Path,
                        default=Path("/cv-corpus-14.0-2023-06-23/"),
                        help="Where Common Voice audio data is saved.")
    parser.add_argument("--flag_dir", type=Path,
                        default=Path("/scratch/as16583/flags"),
                        help="Where flag image files are saved.")
    parser.add_argument("--imagenet_classmapping_filename", type=Path,
                        default=Path("LOC_synset_mapping.txt"),
                        help="ImageNet synset mapping filename (must be .txt).")
    parser.add_argument("--imagenet_dir", type=Path,
                        default=Path("/Users/adrielsaporta/Documents/NYU/symile_data/imagenet/imagenet-object-localization-challenge"),
                        help="Where ImageNet data is saved.")
    parser.add_argument("--imagenet_train_filename", type=Path,
                        default=Path("LOC_train_solution.csv"),
                        help="ImageNet training data filename (must be .csv).")
    parser.add_argument("--wandb", type=str_to_bool, default=False,
                        help="Whether to use wandb for logging.")
    return parser.parse_args()