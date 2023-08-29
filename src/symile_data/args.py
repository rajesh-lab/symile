import argparse
from pathlib import Path

from utils import str_to_bool

def parse_args_generate_data():
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_per_language", type=int, default=25,
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
                        default=Path("/imagenet"),
                        help="Where ImageNet data is saved.")
    parser.add_argument("--imagenet_train_filename", type=Path,
                        default=Path("LOC_train_solution.csv"),
                        help="ImageNet training data filename (must be .csv).")
    parser.add_argument("--save_path", type=str,
                        default="./dataset.csv",
                        help="Where to save dataset csv.")

    return parser.parse_args()

def parse_args_main():
    parser = argparse.ArgumentParser()

    ### DATASET ARGS ###
    parser.add_argument("--dataset_path", type=str,
                        default="./dataset.csv",
                        help="Path to dataset csv.")

    ### MODEL ARGS ###
    parser.add_argument("--audio_model_id", type=str,
                        default="openai/whisper-small",
                        help="Hugging Face model id for audio encoder.")
    parser.add_argument("--image_model_id", type=str,
                        default="openai/clip-vit-base-patch16",
                        help="Hugging Face model id for image encoder.")
    parser.add_argument("--text_model_id", type=str,
                        default="xlm-roberta-base",
                        help="Hugging Face model id for text encoder.")
    parser.add_argument("--d", type=int, default=768,
                        help="Dimensionality used by the linear projection heads \
                              of all three encoders.")
    parser.add_argument("--text_embedding", type=str,
                        choices = ["eos", "bos"], default="eos",
                        help="Whether to use text encoder BOS or EOS embedding \
                              as input to projection head.")

    ### TRAINING ARGS ###
    parser.add_argument("--batch_sz", type=int, default=20,
                        help="Batch size for pretraining.")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of epochs to pretrain for.")
    parser.add_argument("--logit_scale_init", type=float, default=-0.3,
                        help="Value used to initialize the learned logit_scale. \
                              CLIP used np.log(1 / 0.07) = 2.65926.")
    parser.add_argument("--loss_fn", type=str,
                        choices = ["symile", "pairwise_infonce"], default="pairwise_infonce",
                        help="Loss function to use for training.")
    parser.add_argument("--lr", type=float, default=1.0e-1,
                        help="Learning rate.")
    parser.add_argument("--normalize", type=str_to_bool, default=True,
                        help="Whether to normalize representations, both during \
                              pre-training before loss calculation and during evaluation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_seed", type=str_to_bool, default=True,
                        help="Whether to use a seed for reproducibility.")
    parser.add_argument("--wandb", type=str_to_bool, default=False,
                        help="Whether to use wandb for logging.")

    return parser.parse_args()