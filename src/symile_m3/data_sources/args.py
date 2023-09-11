import argparse
from pathlib import Path

from src.utils import str_to_bool


def parse_args_create_image_splits():
    parser = argparse.ArgumentParser()

    parser.add_argument("--imagenet_classmapping_path", type=Path,
                        default=Path("/imagenet/LOC_synset_mapping.txt"),
                        help="Path to ImageNet synset mapping file (must be .txt).")
    parser.add_argument("--imagenet_dir", type=Path,
                        default=Path("/imagenet/ILSVRC/Data/CLS-LOC"),
                        help="Where ImageNet image data is saved.")
    parser.add_argument("--imagenet_train_path", type=Path,
                        default=Path("/imagenet/ILSVRC/ImageSets/CLS-LOC/train_cls.txt"),
                        help="Path to ImageNet training classification data \
                              filename (must be .txt).")
    parser.add_argument("--imagenet_val_path", type=Path,
                        default=Path("/imagenet/LOC_val_solution.csv"),
                        help="Path to ImageNet val classification data \
                              filename (must be .csv).")
    parser.add_argument("--pretrain_train_size", type=float, default=0.8,
                        help="If between 0.0 and 1.0, represents the proportion of \
                              the ImageNet TRAIN dataset to include in the pretrain \
                              train split. If greater than 1.0, must be a a whole \
                              number that represents the absolute number of train \
                              samples.")
    parser.add_argument("--support_train_size", type=float, default=0.7,
                        help="Must be between 0.0 and 1.0. Represents the \
                              proportion of the ImageNet VAL dataset to include \
                              in the support classification train split.")
    parser.add_argument("--support_val_size", type=float, default=0.1,
                        help="Must be between 0.0 and 1.0. Represents the \
                              proportion of the ImageNet VAL dataset to include \
                              in the support classification val split.")
    parser.add_argument("--save_path", type=Path, default=Path("."),
                        help="Where to save dataset csv files.")

    return parser.parse_args()


def parse_args_create_word_splits():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrain_train_size", type=float, default=0.6,
                        help="Must be between 0.0 and 1.0. Represents the \
                              proportion of the words dataset to include in the \
                              pretrain train split.")
    parser.add_argument("--pretrain_val_size", type=float, default=0.1,
                        help="Must be between 0.0 and 1.0. Represents the \
                              proportion of the words dataset to include in the \
                              pretrain val split.")
    parser.add_argument("--support_train_size", type=float, default=0.7,
                        help="Must be between 0.0 and 1.0. Represents the \
                              proportion of the pretrain test words to include \
                              in the support classification train split.")
    parser.add_argument("--support_val_size", type=float, default=0.1,
                        help="Must be between 0.0 and 1.0. Represents the \
                              proportion of the pretrain test words to include \
                              in the support classification val split.")
    parser.add_argument("--save_path", type=Path, default=Path("."),
                        help="Where to save dataset csv files.")
    parser.add_argument("--word_path", type=Path,
                        default=Path("./words.txt"),
                        help="Path to txt file containing word data.")

    return parser.parse_args()


def parse_args_generate_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_per_language", type=int, default=5,
                        help="Number of samples per language per template to \
                              generate.")
    parser.add_argument("--audio_save_dir", type=Path,
                        default=Path("./audio"),
                        help="Where to save generated audio files.")
    parser.add_argument("--commonvoice_dir", type=Path,
                        default=Path("/cv-corpus-14.0-2023-06-23/"),
                        help="Where Common Voice audio data is saved.")
    parser.add_argument("--commonvoice_split", type=str,
                        choices=["train", "dev", "test"],
                        default="train",
                        help="Common Voice split to sample audio data from.")
    parser.add_argument("--flag_dir", type=Path,
                        default=Path("/scratch/as16583/flags"),
                        help="Where flag image files are saved.")
    parser.add_argument("--image_dir", type=Path,
                        default=Path("./img_pretrain_train.csv"),
                        help="Path to csv with image data to sample from.")
    parser.add_argument("--negative_samples", type=str_to_bool, default=False,
                        help="Whether to include negative, along with positive, \
                              samples in a 1:1 ratio.")
    parser.add_argument("--save_path", type=str,
                        default="./dataset_val.csv",
                        help="Where to save dataset csv.")
    parser.add_argument("--word_path", type=Path,
                        default=Path("./words.txt"),
                        help="Path to txt file containing word data.")

    return parser.parse_args()