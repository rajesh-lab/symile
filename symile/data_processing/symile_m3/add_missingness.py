"""
This script takes all the data and add missingness to it where for each sample,
each modality is missing with probability missingness_prob.
"""
import os

import pandas as pd
import torch

from args import parse_args_add_missingness


def add_missingness(args, split, n):
    save_dir = os.path.join(args.save_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    for name in ["text_missingness", "image_missingness", "audio_missingness"]:
        tensor = (torch.rand(n) < args.missingness_prob).int()

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