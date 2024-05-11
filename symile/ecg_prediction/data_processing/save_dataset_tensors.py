import os

import pandas as pd
import torch
from tqdm import tqdm
import wfdb

from args import parse_save_dataset_tensors
from symile.ecg_prediction.constants import GENDER_MAP


def get_ecg(args, pt):
    ecg_pt = args.ecg_data_dir / pt
    signal = torch.from_numpy(wfdb.rdrecord(ecg_pt).p_signal)

    # normalize to be between -1 and 1
    signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1

    return signal.unsqueeze(0).to(torch.float32)


def save_dataset_tensors(args, df, split):
    save_dir = args.data_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ecg_list = []
    age_list = []
    gender_list = []
    study_id_list = []

    n = df.shape[0]
    ecg_tensor = torch.empty((n, 1, 5000, 12))
    age_tensor = torch.empty(n, dtype=torch.int32)
    gender_tensor = torch.empty(n, dtype=torch.int32)
    study_id_tensor = torch.empty(n, dtype=torch.int32)

    for ix, row in tqdm(df.iterrows(), total=n):
        ecg = get_ecg(args, row["path"])
        ecg_tensor[ix] = ecg

        age_tensor[ix] = int(row["age_quantile"])

        gender_tensor[ix] = GENDER_MAP[row["gender"]]

        study_id_tensor[ix] = row["study_id"]

    torch.save(ecg_tensor, save_dir / f"ecg_{split}.pt")
    torch.save(age_tensor, save_dir / f"age_{split}.pt")
    torch.save(gender_tensor, save_dir / f"gender_{split}.pt")
    torch.save(study_id_tensor, save_dir / f"study_id_{split}.pt")


if __name__ == '__main__':
    args = parse_save_dataset_tensors()

    if args.split_to_run == "train":
        print("Saving train tensors...")
        train_df = pd.read_csv(args.data_dir / args.train_csv)
        save_dataset_tensors(args, train_df, "train")
    else:
        val_df = pd.read_csv(args.data_dir / args.val_csv)
        test_df = pd.read_csv(args.data_dir / args.test_csv)

        print("Saving val tensors...")
        save_dataset_tensors(args, val_df, "val")

        print("Saving test tensors...")
        save_dataset_tensors(args, test_df, "test")