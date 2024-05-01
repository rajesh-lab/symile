import json
import os

import pandas as pd
import torch
import torchvision.transforms as t
from tqdm import tqdm
import wfdb

from args import parse_save_dataset_tensors
from symile.risk_prediction.constants import RISK_VECTOR_COLS


def get_ecg(args, pt):
    ecg_pt = args.ecg_data_dir / pt
    signal = torch.from_numpy(wfdb.rdrecord(ecg_pt).p_signal)

    # normalize to be between -1 and 1
    signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1

    return signal.unsqueeze(0).to(torch.float32)


def get_labs(args, row):
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
    assert len(percentiles) == 51, "There should be 51 labs."

    return (torch.tensor(percentiles, dtype=torch.float32),
            torch.tensor(missing_indicators, dtype=torch.int64))


def save_dataset_tensors(args, df, split):
    save_dir = args.data_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ecg_list = []
    labs_percentiles_list = []
    labs_missingness_list = []
    risk_vector_list = []
    hadm_id_list = []

    for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
        ecg = get_ecg(args, row["ecg_path"])
        (labs_percentiles, labs_missingness) = get_labs(args, row)
        risk_vector = torch.tensor(row[RISK_VECTOR_COLS], dtype=torch.float32)

        ecg_list.append(ecg)
        labs_percentiles_list.append(labs_percentiles)
        labs_missingness_list.append(labs_missingness)
        risk_vector_list.append(risk_vector)

        hadm_id_list.append(row["hadm_id"])

    ecg_tensor = torch.stack(ecg_list) # (n_ecg, 1, 5000, 12)
    labs_percentiles_tensor = torch.stack(labs_percentiles_list) # (n, 51)
    labs_missingness_tensor = torch.stack(labs_missingness_list) # (n, 51)
    risk_vector_tensor = torch.stack(risk_vector_list) # (n, 3)
    hadm_id_tensor = torch.tensor(hadm_id_list) # (n,)

    torch.save(ecg_tensor, save_dir / f"ecg_{split}.pt")
    torch.save(labs_percentiles_tensor, save_dir / f"labs_percentiles_{split}.pt")
    torch.save(labs_missingness_tensor, save_dir / f"labs_missingness_{split}.pt")
    torch.save(risk_vector_tensor, save_dir / f"risk_vector_{split}.pt")
    torch.save(hadm_id_tensor, save_dir / f"hadm_id_{split}.pt")


if __name__ == '__main__':
    args = parse_save_dataset_tensors()

    train_df = pd.read_csv(args.data_dir / args.train_csv)
    val_df = pd.read_csv(args.data_dir / args.val_csv)
    test_df = pd.read_csv(args.data_dir / args.test_csv)

    if args.split == "train":
        print("Saving train tensors...")
        save_dataset_tensors(args, train_df, "train")
    elif args.split == "val":
        print("Saving val tensors...")
        save_dataset_tensors(args, val_df, "val")
    elif args.split == "test":
        print("Saving test tensors...")
        save_dataset_tensors(args, test_df, "test")

    if args.split == "train":
        with open(args.data_dir / "metadata.json", 'w') as f:
            json.dump({
                "train set size": len(train_df),
                "val set size": len(val_df),
                "test set size": len(test_df)
            }, f, indent=4)