import argparse
from pathlib import Path


def parse_args_generate_data():
    parser = argparse.ArgumentParser()

    ### IMAGENET ARGS ###
    parser.add_argument("--imagenet_dir", type=Path,
                        default=Path("/gpfs/data/ranganathlab/adriel/imagenet"),
                        help="Directory where _all_ ImageNet files are held.")
    parser.add_argument("--imagenet_classmapping_path", type=Path,
                        default=Path("LOC_synset_mapping.txt"),
                        help="Path to ImageNet synset mapping file (must be .txt).")
    parser.add_argument("--imagenet_image_train_data_dir", type=Path,
                        default=Path("ILSVRC/Data/CLS-LOC/train"),
                        help="Directory where ImageNet image train data is held.")
    parser.add_argument("--imagenet_train_cls_path", type=Path,
                        default=Path("ILSVRC/ImageSets/CLS-LOC/train_cls.txt"),
                        help="Path to ImageNet training classification data \
                              filename (must be .txt).")

    ### WORD ARGS ###
    parser.add_argument("--word_path", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/symile_m3/data/sources/words.txt"),
                        help="Path to txt file containing word data.")

    ### SYMILE ARGS ###
    parser.add_argument("--audio_save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/audio"),
                        help="Where to save generated audio files.")
    parser.add_argument("--pretrain_n", type=int, default=6,
                        help="Number of samples for combined pretrain train and \
                              val sets.")
    parser.add_argument("--pretrain_val_size", type=float, default=0.5,
                        help="Should be between 0.0 and 1.0. Represents the \
                              proportion of pretrain data to include in val split.")
    parser.add_argument("--save_dir", type=Path,
                        default=Path("/gpfs/scratch/as16583/symile/src/symile_m3/data/sources"),
                        help="Directory to save dataset csvs in.")
    parser.add_argument("--support_n", type=int, default=4,
                        help="Number of _positive_ samples for combined support \
                              train/val/test sets. Note that negative samples are \
                              generated from the positive samples.")
    parser.add_argument("--support_train_size", type=float, default=0.7,
                        help="Must be between 0.0 and 1.0. Represents the \
                              proportion of the support dataset to include \
                              in the support classification train split.")
    parser.add_argument("--support_val_size", type=float, default=0.1,
                        help="Must be between 0.0 and 1.0. Represents the \
                              proportion of the support dataset to include \
                              in the support classification val split.")
    parser.add_argument("--zeroshot_n", type=int, default=4,
                        help="Number of samples for zeroshot test set.")

    return parser.parse_args()