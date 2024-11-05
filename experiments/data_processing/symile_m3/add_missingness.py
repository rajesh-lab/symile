"""
This script applies missingness to the Symile-M3 dataset by generating and
saving missingness indicator tensors for each modality (text, image, audio).
The resulting missingness indicators are saved as .pt files in split-specific
directories.
"""
import os

import torch

from args import parse_args_add_missingness


def add_missingness(args, split, n):
    """
    Generates and saves missingness tensors for text, image, and audio based on
    a specified probability. Each tensor has a length of `n` and is created by
    drawing samples from a Bernoulli distribution, where each element is set to
    1 (missing) with probability `args.missingness_prob`, and 0 otherwise.

    Args:
        args: Argument object containing the save directory and missingness probability.
        split (str): The dataset split ("train" or "val") to determine the save path.
        n (int): The length of the missingness tensor to generate.
    """
    save_dir = os.path.join(args.save_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    for name in ["text_missingness", "image_missingness", "audio_missingness"]:
        tensor = torch.bernoulli(torch.full((n,), args.missingness_prob)).int()

        missingness_str = f"{args.missingness_prob:.2f}"[2:]

        save_pt = os.path.join(save_dir, f"{name}_prob{missingness_str}_{split}.pt")

        torch.save(tensor, save_pt)

        print(f"Saved {name} for {split} with length {len(tensor)}.")


if __name__ == '__main__':
    args = parse_args_add_missingness()

    for split in ["train", "val"]:
        split_dir = args.data_dir / split
        idx = torch.load(split_dir / f"idx_{split}.pt")
        add_missingness(args, split, len(idx))