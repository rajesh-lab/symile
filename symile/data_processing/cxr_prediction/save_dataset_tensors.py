import json
import os

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as t
from tqdm import tqdm
import wfdb

from args import parse_save_dataset_tensors
from symile.cxr_prediction.constants import IMAGENET_MEAN, IMAGENET_STD


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


def save_dataset_tensors(args, df, split):
    save_dir = args.data_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

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

        if split in ["val_query", "val_candidate", "test_query", "test_candidate"]:
            label_name_list.append(row["label_name"])
            label_value_list.append(row["label_value"])

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

    if split in ["val_query", "val_candidate", "test_query", "test_candidate"]:
        label_value_tensor = torch.tensor(label_value_list)
        torch.save(label_value_tensor, save_dir / f"label_value_{split}.pt")

        with open(f"{save_dir}/label_name_{split}.txt", 'w') as f:
            f.writelines("\n".join(label_name_list))


if __name__ == '__main__':
    args = parse_save_dataset_tensors()

    train_df = pd.read_csv(args.data_dir / args.train_csv)

    val_query_df = pd.read_csv(args.data_dir / args.val_query_csv)
    val_candidate_df = pd.read_csv(args.data_dir / args.val_candidate_csv)
    val_df = pd.read_csv(args.data_dir / args.val_csv)

    test_query_df = pd.read_csv(args.data_dir / args.test_query_csv)
    test_candidate_df = pd.read_csv(args.data_dir / args.test_candidate_csv)
    # full test set is the candidate set without duplicates
    test_df = test_candidate_df.drop_duplicates(subset=["hadm_id"])
    test_df.to_csv(args.data_dir / "test.csv", index=False)

    print("Saving train tensors...")
    save_dataset_tensors(args, train_df, "train")

    print("Saving val query tensors...")
    save_dataset_tensors(args, val_query_df, "val_query")

    print("Saving val candidate tensors...")
    save_dataset_tensors(args, val_candidate_df, "val_candidate")

    print("Saving val tensors...")
    save_dataset_tensors(args, val_df, "val")

    print("Saving test query tensors...")
    save_dataset_tensors(args, test_query_df, "test_query")

    print("Saving test candidate tensors...")
    save_dataset_tensors(args, test_candidate_df, "test_candidate")

    print("Saving test tensors...")
    save_dataset_tensors(args, test_df, "test")

    with open(args.data_dir / "metadata.json", "w") as f:
        json.dump({
            "train set size": len(train_df),
            "val set size": len(val_df),
            "val query set size": len(val_query_df),
            "val candidate set size": len(val_candidate_df),
            "test set size": len(test_df),
            "test query set size": len(test_query_df),
            "test candidate set size": len(test_candidate_df),
            "cxr_scale": args.cxr_scale,
            "cxr_crop": args.cxr_crop
        }, f, indent=4)