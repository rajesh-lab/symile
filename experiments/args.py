"""
Defines functions for processing command-line arguments for main.py, test.py,
and symile/experiments/data_processing/binary_xor/informations.py.
"""
import argparse
from pathlib import Path

from utils import str_to_bool


def parse_args_informations():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_v", type=int, default=2,
                        help="Dimensionality of binary vectors.")

    parser.add_argument("--save_dir", type=Path,
                        help="Where to save information results.")

    return parser.parse_args()


def parse_args_main():
    """
    Parses command-line arguments for main.py.

    First parses the `--experiment` argument to determine which experiment is being run.
    Common arguments applicable to both experiments are included, and additional
    arguments based on the value of `--experiment` are conditionally added and parsed.

    Returns:
        argparse.Namespace: A namespace object containing the parsed arguments.
    """
    # first parse only the --experiment argument
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", type=str,
                        choices=["binary_xor", "symile_m3", "symile_mimic"],
                        required=True,
                        help="Which experiment is being run.")
    args, remaining_argv = parser.parse_known_args()

    # create the main parser
    parser = argparse.ArgumentParser()

    ### ARGUMENTS COMMON TO ALL EXPERIMENTS ###
    parser.add_argument("--batch_sz_train", type=int,
                        help="Train batch size for pretraining.")
    parser.add_argument("--batch_sz_val", type=int,
                        help="Val set batch size for pretraining.")
    parser.add_argument("--batch_sz_test", type=int,
                        help="Test set batch size.")
    parser.add_argument("--check_val_every_n_epoch", type=int,
                        help="Check val every n train epochs.")
    parser.add_argument("--ckpt_save_dir", type=Path,
                        help="Where to save model checkpoints.")
    parser.add_argument("--d", type=int,
                        help="Dimensionality used by the linear projection heads \
                              of all three encoders.")
    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")
    parser.add_argument("--drop_last", type=str_to_bool,
                        help="Whether to drop the last non-full batch of each \
                              DataLoader worker's dataset replica.")
    parser.add_argument("--negative_sampling", type=str,
                        choices = ["n", "n_squared"],
                        help="We explore two variants for negative sampling within \
                              a batch of n samples: `n` [for O(n)] draws n - 1 \
                              negative samples for each positive, `n_squared` \
                              [for O(n^2)] draws n^2 - 1 negative samples for each \
                              positive.")
    parser.add_argument("--epochs", type=int,
                        help="Number of epochs to pretrain for.")
    parser.add_argument("--freeze_logit_scale", type=str_to_bool, default=False,
                        help="Whether to freeze logit scale during pretraining.")
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Path of the checkpoint from which training is resumed.")
    parser.add_argument("--logit_scale_init", type=float,
                        help="Value used to initialize the learned logit_scale. \
                              CLIP used np.log(1 / 0.07) = 2.65926.")
    parser.add_argument("--loss_fn", type=str, default="symile",
                        choices = ["symile", "clip"],
                        help="Loss function to use for training.")
    parser.add_argument("--lr", type=float,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay coefficient used by AdamW optimizer.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=False,
                        help="Whether to use a seed for reproducibility.")
    parser.add_argument("--wandb", type=str_to_bool, default=False,
                        help="Whether to use wandb for logging.")
    parser.add_argument("--wandb_run_id", type=str,
                        default=None,
                        help="Use if loading from checkpoint and using WandbLogger.")
    # debugging args
    parser.add_argument("--limit_train_batches", type=float, default=1.0,
                        help="How much of training dataset to check. Useful \
                              when debugging. 1.0 is default used by Trainer. \
                              Set to 0.1 to check 10% of dataset.")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="How much of val dataset to check. Useful \
                              when debugging. 1.0 is default used by Trainer. \
                              Set to 0.1 to check 10% of dataset.")

    ### BINARY XOR ARGS ###
    if args.experiment == "binary_xor":
        parser.add_argument("--train_n", type=int,
                            help="Number of samples (a, b, c) in train dataset.")
        parser.add_argument("--val_n", type=int,
                            help="Number of samples (a, b, c) in val dataset.")
        parser.add_argument("--test_n", type=int,
                            help="Number of samples (a, b, c) in test dataset.")
        parser.add_argument("--d_v", type=int,
                            help="Dimensionality of dataset vectors.")
        parser.add_argument("--bootstrap", type=str_to_bool, default=False,
                        help="Whether to bootstrap test results.")
        parser.add_argument("--bootstrap_n", type=int, default=10,
                        help="Number of bootstrap samples.")
    ### SYMILE-M3 ARGS ###
    elif args.experiment == "symile_m3":
        # data args
        parser.add_argument("--train_csv", type=Path,
                                default=Path("train.csv"),
                                help="Filename for train csv.")
        parser.add_argument("--val_csv", type=Path,
                                default=Path("val.csv"),
                                help="Filename for val csv.")
        parser.add_argument("--test_csv", type=Path,
                                default=Path("test.csv"),
                                help="Filename for test csv.")
        parser.add_argument("--translations_path", type=Path,
                                help="Path to json file with ImageNet class names, \
                                      synset ids, and translations.")
        parser.add_argument("--num_langs", type=int,
                                help="Number of languages in generated text.")
        parser.add_argument("--missingness", type=str_to_bool, default=False,
                                help="Whether to train with missingness.")
        parser.add_argument("--missingness_prob", type=float,
                                help="Probability with which a given modality is missing.")
        # model args
        parser.add_argument("--audio_model_id", type=str,
                            help="Hugging Face model id for audio encoder.")
        parser.add_argument("--image_model_id", type=str,
                            help="Hugging Face model id for image encoder.")
        parser.add_argument("--text_model_id", type=str,
                            help="Hugging Face model id for text encoder.")
        parser.add_argument("--text_embedding", type=str,
                                choices = ["eos", "bos"], default="eos",
                                help="Whether to use text encoder BOS or EOS embedding \
                                    as input to projection head.")
        parser.add_argument("--metadata_filename", type=Path,
                                default=Path("metadata.json"),
                                help="Path to json file with metadata for all encoders.")
    ### SYMILE-MIMIC ARGS ###
    elif args.experiment == "symile_mimic":
        parser.add_argument("--pretrained", type=str_to_bool, default=False,
                            help="Whether to pretrained encoders for CXR and ECG.")

    all_args = parser.parse_args(remaining_argv)

    # manually set the --experiment argument
    all_args.experiment = args.experiment

    return all_args


def parse_args_test():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str,
                        choices = ["symile_m3", "symile_mimic"],
                        help="Which experiment is being run.")

    ### ARGUMENTS COMMON TO BOTH EXPERIMENTS ###
    parser.add_argument("--batch_sz_test", type=int,
                        help="Test set batch size.")
    parser.add_argument("--bootstrap", type=str_to_bool, default=False,
                        help="Whether to bootstrap test results.")
    parser.add_argument("--bootstrap_n", type=int, default=10,
                        help="Number of bootstrap samples.")
    parser.add_argument("--data_dir", type=Path,
                        help="Directory with dataset csvs.")
    parser.add_argument("--description" , type=str, default="",
                        help="Description of the test run.")
    parser.add_argument("--ckpt_path", type=str,
                        help="Path of the checkpoint to use.")
    parser.add_argument("--save_dir", type=Path,
                        help="Where to save test results.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")

    ### SYMILE-M3 ARGS ###
    parser.add_argument("--num_langs", type=int,
                        help="Number of languages in generated text.")

    return parser.parse_args()


def parse_args_collect_tuning_results():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str,
                        choices = ["symile_m3", "symile_mimic"], required=True,
                        help="Which experiment is being run.")
    parser.add_argument("--results_pt", type=Path,
                        help="Path to yaml file with hyperparameter tuning results.")
    parser.add_argument("--save_pt", type=Path,
                        help="Where to save test results.")

    return parser.parse_args()