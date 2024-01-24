import argparse
from pathlib import Path

from src.utils import str_to_bool


def parse_args_source_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_reference", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/high_dim/data_reference.json"),
                        help="Path to json file with class names, ImageNet synset \
                              id, and language translations.")
    parser.add_argument("--imagenet_classmapping_path", type=Path,
                        default=Path("/gpfs/data/ranganathlab/imagenet/LOC_synset_mapping.txt"),
                        help="Path to ImageNet synset mapping txt file.")

    return parser.parse_args()


def parse_args_generate_data():
    parser = argparse.ArgumentParser()

    ### DATA ARGS ###
    parser.add_argument("--data_type", type=str,
                        choices = ["overlap", "disjoint"], default="disjoint",
                        help="Whether to allow overlap across languauge and \
                              meaning (overlap) or not (disjoint).")
    parser.add_argument("--text_len", type=int, default=2,
                        help="Number of words in generated text.")
    parser.add_argument("--cv_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/adriel/cv/cv"),
                        help="Directory where CommonVoice audio clips are held.")
    parser.add_argument("--data_reference", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/high_dim/data_reference.json"),
                        help="Path to json file with class names, ImageNet synset \
                              id, and language translations.")
    parser.add_argument("--imagenet_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/imagenet/ILSVRC/Data/CLS-LOC/train"),
                        help="Directory where ImageNet image train data is held.")

    ### SYMILE ARGS ###
    parser.add_argument("--pretrain_n", type=int, default=6,
                        help="Number of samples for combined pretrain train and \
                              val sets.")
    parser.add_argument("--val_size", type=float, default=0.5,
                        help="Should be between 0.0 and 1.0. Represents the \
                              proportion of pretrain data to include in val split.")
    parser.add_argument("--test_n", type=int, default=4,
                        help="Number of samples for zeroshot test set.")
    parser.add_argument("--save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/high_dim/data/data_c3_l3_t3"),
                        help="Directory to save dataset csvs in.")

    return parser.parse_args()


def parse_args_save_representations():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/high_dim/datasets/t5/disjoint"),
                        help="Directory with dataset csvs.")
    parser.add_argument("--train_csv", type=Path,
                        default=Path("train.csv"),
                        help="Filename for train csv.")
    parser.add_argument("--val_csv", type=Path,
                        default=Path("val.csv"),
                        help="Filename for val csv.")
    parser.add_argument("--test_csv", type=Path,
                        default=Path("zeroshot.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/high_dim/datasets/t5/disjoint"),
                        help="Directory to save dataset tensors in.")

    ### MODEL ARGS ###
    parser.add_argument("--audio_model_id", type=str,
                        default="openai/whisper-tiny",
                        choices = ["openai/whisper-small", "openai/whisper-tiny"],
                        help="Hugging Face model id for audio encoder.")
    parser.add_argument("--image_model_id", type=str,
                        default="openai/clip-vit-base-patch16",
                        help="Hugging Face model id for image encoder.")
    parser.add_argument("--text_model_id", type=str,
                        default="bert-base-multilingual-cased",
                        choices = ["bert-base-multilingual-cased", "xlm-roberta-base"],
                        help="Hugging Face model id for text encoder.")

    ### TRAINING ARGS ###
    parser.add_argument("--batch_sz_train", type=int, default=256,
                        help="Train batch size for pretraining.")
    parser.add_argument("--batch_sz_val", type=int, default=256,
                        help="Val set batch size for pretraining.")
    parser.add_argument("--batch_sz_test", type=int, default=256,
                        help="Test set batch size.")
    parser.add_argument("--drop_last", type=str_to_bool, default=False,
                        help="Whether to drop the last non-full batch of each \
                              DataLoader worker's dataset replica.")

    return parser.parse_args()