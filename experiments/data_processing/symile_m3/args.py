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


def parse_args_generate_translations():
    parser = argparse.ArgumentParser()

    parser.add_argument("--translations_path", type=Path,
                        help="Path to json file with ImageNet class names, \
                              synset ids, and translations.")
    parser.add_argument("--imagenet_classmapping_path", type=Path,
                        help="Path to ImageNet synset mapping txt file.")
    parser.add_argument("--manual_translations_path", type=Path,
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
                        help="Directory where CommonVoice audio clips are held.")
    parser.add_argument("--translations_path", type=Path,
                        help="Path to json file with ImageNet class names, \
                              synset ids, and translations.")
    parser.add_argument("--imagenet_dir", type=Path,
                        help="Directory where ImageNet image train data is held.")

    ### SYMILE ARGS ###
    parser.add_argument("--train_n", type=int,
                        help="Number of samples for train set.")
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
                        choices = ["all", "train", "val", "test"])
    parser.add_argument("--cv_dir", type=Path,
                        help="Directory where CommonVoice audio clips are held.")
    parser.add_argument("--imagenet_dir", type=Path,
                        help="Directory where ImageNet image train data is held.")

    ### MODEL ARGS ###
    parser.add_argument("--audio_model_id", type=str,
                        help="Hugging Face model id for audio encoder.")
    parser.add_argument("--image_model_id", type=str,
                        help="Hugging Face model id for image encoder.")
    parser.add_argument("--text_model_id", type=str,
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


def parse_args_add_missingness():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=Path,
                        help="Directory with subdirectories.")
    parser.add_argument("--save_dir", type=Path,
                        help="Directory to save merged tensors in.")
    parser.add_argument("--missingness_prob", type=float, default=0.5,
                        help="Probability with which a given modality is missing.")

    return parser.parse_args()