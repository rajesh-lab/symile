"""
This script loads and preprocesses the Symile-MIMIC dataset splits, saving the
resulting tensors to split-specific directories in `data_dir`.
"""
import json
import os
import time

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as t
from tqdm import tqdm
import wfdb

from args import parse_process_and_save_tensors
from constants import IMAGENET_MEAN, IMAGENET_STD


def get_cxr(args, pt, split):
    """
    Loads and preprocesses a chest X-ray (CXR) image.
    """
    cxr_pt = args.cxr_data_dir / pt
    img = Image.open(cxr_pt).convert('RGB')

    # square crop
    if split == "train":
        crop = t.RandomCrop((args.cxr_crop, args.cxr_crop))
    else:
        crop = t.CenterCrop((args.cxr_crop, args.cxr_crop))

    transform = t.Compose([
        # smaller edge is scaled to `cxr_scale`. i.e, if height > width,
        # then img is rescaled to (cxr_scale * height / width, cxr_scale)
        t.Resize(args.cxr_scale),
        crop,
        t.ToTensor(),
        t.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(img)


def get_ecg(args, pt):
    """
    Loads and preprocesses an ECG signal.
    """
    ecg_pt = args.ecg_data_dir / pt
    signal = torch.from_numpy(wfdb.rdrecord(ecg_pt).p_signal)

    # normalize to be between -1 and 1
    signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1

    return signal.unsqueeze(0).to(torch.float32)


def get_labs(args, row):
    """
    Processes laboratory data for a given row, handling missing values and
    ensuring consistent order.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two tensors, one for the lab percentiles
                                           and one for the missing indicators.
    """
    percentiles = []
    missing_indicators = []

    labs_means = json.load(open(args.data_dir / args.labs_means))

    for col_p in sorted(labs_means.keys()): # sort to ensure order is consistent
        col = col_p.replace("_percentile", "")

        if pd.isna(row.get(col)):
            # lab is missing
            percentiles.append(labs_means[col_p])
            missing_indicators.append(0)
        else:
            # lab is not missing
            percentiles.append(row[col_p])
            missing_indicators.append(1)

    assert len(percentiles) == len(missing_indicators), \
        "Lengths of percentiles and missing indicators must match."
    assert len(percentiles) == 50, "There should be 50 labs."

    return (torch.tensor(percentiles, dtype=torch.float32),
            torch.tensor(missing_indicators, dtype=torch.int64))


def process_and_save_tensors(args, df, split):
    """
    Processes the data from df and saves the resulting tensors to the specified
    directory.
    """
    save_dir = args.data_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cxr_list = []
    ecg_list = []
    labs_percentiles_list = []
    labs_missingness_list = []
    hadm_id_list = []
    label_hadm_id_list = []
    label_list = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        cxr = get_cxr(args, row["cxr_path"], split)
        ecg = get_ecg(args, row["ecg_path"])
        (labs_percentiles, labs_missingness) = get_labs(args, row)

        cxr_list.append(cxr)
        ecg_list.append(ecg)
        labs_percentiles_list.append(labs_percentiles)
        labs_missingness_list.append(labs_missingness)
        hadm_id_list.append(row["hadm_id"])

        if split in ["val_retrieval", "test"]:
            label_hadm_id_list.append(row["label_hadm_id"])
            label_list.append(row["label"])

    cxr_tensor = torch.stack(cxr_list) # (n, 3, cxr_crop, cxr_crop)
    ecg_tensor = torch.stack(ecg_list) # (n, 1, 5000, 12)
    labs_percentiles_tensor = torch.stack(labs_percentiles_list) # (n, 50)
    labs_missingness_tensor = torch.stack(labs_missingness_list) # (n, 50)
    hadm_id_tensor = torch.tensor(hadm_id_list) # (n,)

    torch.save(cxr_tensor, save_dir / f"cxr_{split}.pt")
    torch.save(ecg_tensor, save_dir / f"ecg_{split}.pt")
    torch.save(labs_percentiles_tensor, save_dir / f"labs_percentiles_{split}.pt")
    torch.save(labs_missingness_tensor, save_dir / f"labs_missingness_{split}.pt")
    torch.save(hadm_id_tensor, save_dir / f"hadm_id_{split}.pt")

    if split in ["val_retrieval", "test"]:
        label_hadm_id_tensor = torch.tensor(label_hadm_id_list)
        torch.save(label_hadm_id_tensor, save_dir / f"label_hadm_id_{split}.pt")

        label_tensor = torch.tensor(label_list)
        torch.save(label_tensor, save_dir / f"label_{split}.pt")


if __name__ == '__main__':
    start = time.time()

    args = parse_process_and_save_tensors()

    train_df = pd.read_csv(args.data_dir / args.train_csv)
    val_df = pd.read_csv(args.data_dir / args.val_csv)
    val_retrieval_df = pd.read_csv(args.data_dir / args.val_retrieval_csv)
    test_df = pd.read_csv(args.data_dir / args.test_csv)

    print("Saving train tensors...")
    process_and_save_tensors(args, train_df, "train")

    print("Saving val tensors...")
    process_and_save_tensors(args, val_df, "val")

    print("Saving val retrieval tensors...")
    process_and_save_tensors(args, val_retrieval_df, "val_retrieval")

    print("Saving test tensors...")
    process_and_save_tensors(args, test_df, "test")

    with open(args.data_dir / "metadata.json", "w") as f:
        json.dump({
            "train size": len(train_df),
            "val size": len(val_df),
            "val retrieval size": len(val_retrieval_df),
            "test size": len(test_df),
            "cxr_scale": args.cxr_scale,
            "cxr_crop": args.cxr_crop
        }, f, indent=4)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")