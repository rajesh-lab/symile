import argparse
from pathlib import Path

from symile.utils import str_to_bool


def parse_args_source_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_reference", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/symile/high_dim/datasets/data_reference.json"),
                        help="Path to json file with class names, ImageNet synset \
                              id, and language translations.")
    parser.add_argument("--imagenet_classmapping_path", type=Path,
                        default=Path("/gpfs/data/ranganathlab/imagenet/LOC_synset_mapping.txt"),
                        help="Path to ImageNet synset mapping txt file.")
    parser.add_argument("--manual_translations_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/symile/high_dim/datasets/manual_translations.json"),
                        help="Path to manual translations json file.")

    return parser.parse_args()


def parse_args_generate_data():
    parser = argparse.ArgumentParser()

    ### DATA ARGS ###
    parser.add_argument("--data_type", type=str,
                        choices = ["overlap", "disjoint"], default="disjoint",
                        help="Whether to allow overlap across languauge and \
                              meaning (overlap) or not (disjoint).")
    parser.add_argument("--num_words", type=int, default=5,
                        help="Number of words in generated text.")
    parser.add_argument("--num_langs", type=int, default=5,
                        help="Number of languages in generated text.")
    parser.add_argument("--cv_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/commonvoice/cv"),
                        help="Directory where CommonVoice audio clips are held.")
    parser.add_argument("--data_reference", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/high_dim/datasets/data_reference.json"),
                        help="Path to json file with class names, ImageNet synset \
                              id, and language translations.")
    parser.add_argument("--imagenet_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/imagenet"),
                        help="Directory where ImageNet image train data is held.")

    ### SYMILE ARGS ###
    parser.add_argument("--dataset_n", type=int,
                        help="Number of samples for combined train, val, and \
                              test sets.")
    parser.add_argument("--val_n", type=int,
                        help="Number of samples for val set.")
    parser.add_argument("--test_n", type=int,
                        help="Number of samples for test set.")
    parser.add_argument("--save_dir", type=Path,
                        help="Directory to save dataset csvs in.")

    return parser.parse_args()


def parse_args_split_df():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_to_split", type=Path)
    parser.add_argument("--sub_df_size", type=int, default=500000,
                        help="Number of rows for each sub-df.")
    parser.add_argument("--save_dir", type=Path)

    return parser.parse_args()


def parse_args_max_token_len():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path)
    parser.add_argument("--train_csv", type=Path,
                        default=Path("train.csv"),
                        help="Filename for train csv.")
    parser.add_argument("--val_csv", type=Path,
                        default=Path("val.csv"),
                        help="Filename for val csv.")
    parser.add_argument("--test_csv", type=Path,
                        default=Path("test.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--save_pt", type=Path,
                        help="Path to save json file with max token lengths.")
    parser.add_argument("--text_model_id", type=str,
                        help="Hugging Face model id for text tokenizer.")

    return parser.parse_args()


def parse_args_save_representations():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")
    parser.add_argument("--train_csv", type=Path,
                        default=Path("train.csv"),
                        help="Filename for train csv.")
    parser.add_argument("--val_csv", type=Path,
                        default=Path("val.csv"),
                        help="Filename for val csv.")
    parser.add_argument("--test_csv", type=Path,
                        default=Path("test.csv"),
                        help="Filename for test csv.")
    parser.add_argument("--max_token_len_pt", type=Path,
                        help="Path to json file with max token lengths.")
    parser.add_argument("--save_dir", type=Path,
                        help="Directory to save dataset tensors in.")
    parser.add_argument("--split_to_run", type=str,
                        choices = ["all", "train", "val_test"])
    parser.add_argument("--cv_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/commonvoice/cv"),
                        help="Directory where CommonVoice audio clips are held.")
    parser.add_argument("--imagenet_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/imagenet"),
                        help="Directory where ImageNet image train data is held.")

    ### MODEL ARGS ###
    parser.add_argument("--audio_model_id", type=str,
                        default="openai/whisper-tiny",
                        help="Hugging Face model id for audio encoder.")
    parser.add_argument("--image_model_id", type=str,
                        default="openai/clip-vit-base-patch16",
                        help="Hugging Face model id for image encoder.")
    parser.add_argument("--text_model_id", type=str,
                        default="bert-base-multilingual-cased",
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


def parse_args_merge_representations():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path,
                        help="Directory with subdirectories.")
    parser.add_argument("--save_dir", type=Path,
                        help="Directory to save merged tensors in.")
    parser.add_argument("--num_subdirs", type=int,
                        help="Number of subdirectories.")

    return parser.parse_args()