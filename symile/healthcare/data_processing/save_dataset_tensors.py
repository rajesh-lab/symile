import json

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as t
from tqdm import tqdm
import wfdb

from src.healthcare.args import parse_save_dataset_tensors
from src.healthcare.constants import *


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
        t.ToTensor(),
        t.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform(img)


def get_ecg(args, pt):
    ecg_pt = args.ecg_data_dir / pt
    signal = torch.from_numpy(wfdb.rdrecord(ecg_pt).p_signal)

    # normalize to be between -1 and 1
    signal = 2 * (signal - signal.min()) / (signal.max() - signal.min()) - 1

    return signal.unsqueeze(0).to(torch.float32)


def get_labs(args, row):
    percentiles = []
    missing_indicators = []

    for col_p in sorted(LABS_MEANS.keys()): # ensures order is consistent
        col = col_p.replace("_percentile", "")

        if pd.isna(row.get(col)):
            # lab is missing
            percentiles.append(LABS_MEANS[col_p])
            missing_indicators.append(0)
        else:
            # lab is not missing
            percentiles.append(row[col_p])
            missing_indicators.append(1)

    assert len(percentiles) == len(missing_indicators), "Lengths of percentiles and missing indicators must match."
    assert len(percentiles) == 50, "There should be 50 labs."

    return (torch.tensor(percentiles, dtype=torch.float32), torch.tensor(missing_indicators, dtype=torch.int64))


def save_dataset_tensors(args, df, split):
    cxr_list = []
    ecg_list = []
    labs_percentiles_list = []
    labs_missingness_list = []
    hadm_id_list = []
    label_name_list = []
    label_value_list = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        cxr = get_cxr(args, row["cxr_path"], split)
        ecg = get_ecg(args, row["ecg_path"])
        (labs_percentiles, labs_missingness) = get_labs(args, row)

        cxr_list.append(cxr)
        ecg_list.append(ecg)
        labs_percentiles_list.append(labs_percentiles)
        labs_missingness_list.append(labs_missingness)
        hadm_id_list.append(row["hadm_id"])

        if split in ["query", "candidate", "query_val", "candidate_val"]:
            label_name_list.append(row["label_name"])
            label_value_list.append(row["label_value"])

    cxr_tensor = torch.stack(cxr_list) # (n, 3, cxr_crop, cxr_crop)
    ecg_tensor = torch.stack(ecg_list) # (n, 1, 5000, 12)
    labs_percentiles_tensor = torch.stack(labs_percentiles_list) # (n, 50)
    labs_missingness_tensor = torch.stack(labs_missingness_list) # (n, 50)
    hadm_id_tensor = torch.tensor(hadm_id_list) # (n,)

    torch.save(cxr_tensor, args.data_dir / f"cxr_{split}.pt")
    torch.save(ecg_tensor, args.data_dir / f"ecg_{split}.pt")
    torch.save(labs_percentiles_tensor, args.data_dir / f"labs_percentiles_{split}.pt")
    torch.save(labs_missingness_tensor, args.data_dir / f"labs_missingness_{split}.pt")
    torch.save(hadm_id_tensor, args.data_dir / f"hadm_id_{split}.pt")

    if split in ["query", "candidate", "query_val", "candidate_val"]:
        label_value_tensor = torch.tensor(label_value_list)
        torch.save(label_value_tensor, args.data_dir / f"label_value_{split}.pt")

        with open(f"{args.data_dir}/label_name_{split}.txt", 'w') as f:
            f.writelines("\n".join(label_name_list))


if __name__ == '__main__':
    args = parse_save_dataset_tensors()

    train_df = pd.read_csv(args.data_dir / args.train_csv)
    val_df = pd.read_csv(args.data_dir / args.val_csv)
    query_df = pd.read_csv(args.data_dir / args.query_csv)
    candidate_df = pd.read_csv(args.data_dir / args.candidate_csv)
    query_val_df = pd.read_csv(args.data_dir / args.query_val_csv)
    candidate_val_df = pd.read_csv(args.data_dir / args.candidate_val_csv)

    print("Saving train tensors...")
    save_dataset_tensors(args, train_df, "train")

    print("Saving val tensors...")
    save_dataset_tensors(args, val_df, "val")

    print("Saving test query tensors...")
    save_dataset_tensors(args, query_df, "query")

    print("Saving test candidate tensors...")
    save_dataset_tensors(args, candidate_df, "candidate")

    print("Saving val query tensors...")
    save_dataset_tensors(args, query_val_df, "query_val")

    print("Saving val candidate tensors...")
    save_dataset_tensors(args, candidate_val_df, "candidate_val")

    with open(args.data_dir / "metadata.json", 'w') as f:
        json.dump({
            "train set size": len(train_df),
            "val set size": len(val_df),
            "query set size": len(query_df),
            "candidate set size": len(candidate_df),
            "query val set size": len(query_val_df),
            "candidate val set size": len(candidate_val_df),
            "cxr_scale": args.cxr_scale,
            "cxr_crop": args.cxr_crop
        }, f, indent=4)