"""with missigness because it saves cxrs as ints"""
import json
import os

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as t
from tqdm import tqdm
import wfdb

from symile.risk_prediction.args import parse_save_dataset_tensors
from symile.risk_prediction.constants import RISK_VECTOR_COLS


def get_cxr(args, pt, split):
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
        t.ToTensor()
    ])

    cxr = transform(img)
    cxr = (cxr * 255)
    assert torch.all(cxr == cxr.long()), "cxr tensor is not integer-valued."
    return cxr.to(torch.uint8)


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

    cxr_list = []
    ecg_list = []
    risk_vector_list = []
    hadm_id_list = []

    cxr_mapping = {}
    ecg_mapping = {}

    for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
        hadm_id = row["hadm_id"]

        if pd.notnull(row["cxr_path"]):
            cxr = get_cxr(args, row["cxr_path"], split)
            cxr_list.append(cxr)
            cxr_mapping[hadm_id] = len(cxr_list) - 1

        if pd.notnull(row["ecg_path"]):
            ecg = get_ecg(args, row["ecg_path"])
            ecg_list.append(ecg)
            ecg_mapping[hadm_id] = len(ecg_list) - 1

        risk_vector = torch.tensor(row[RISK_VECTOR_COLS], dtype=torch.float32)
        risk_vector_list.append(risk_vector)

        hadm_id_list.append(row["hadm_id"])

        if split in ["query", "candidate", "query_val", "candidate_val"]:
            label_name_list.append(row["label_name"])
            label_value_list.append(row["label_value"])

    cxr_tensor = torch.stack(cxr_list) # (n_cxr, 3, cxr_crop, cxr_crop)
    ecg_tensor = torch.stack(ecg_list) # (n_ecg, 1, 5000, 12)
    risk_vector_tensor = torch.stack(risk_vector_list) # (n, 3)
    hadm_id_tensor = torch.tensor(hadm_id_list) # (n,)

    torch.save(cxr_tensor, save_dir / f"cxr_{split}.pt")
    torch.save(ecg_tensor, save_dir / f"ecg_{split}.pt")
    torch.save(risk_vector_tensor, save_dir / f"risk_vector_{split}.pt")
    torch.save(hadm_id_tensor, save_dir / f"hadm_id_{split}.pt")

    with open(save_dir / f"cxr_mapping_{split}.json", "w") as f:
        json.dump(cxr_mapping, f, indent=4)

    with open(save_dir / f"ecg_mapping_{split}.json", "w") as f:
        json.dump(ecg_mapping, f, indent=4)


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
                "test set size": len(test_df),
                "cxr_scale": args.cxr_scale,
                "cxr_crop": args.cxr_crop
            }, f, indent=4)