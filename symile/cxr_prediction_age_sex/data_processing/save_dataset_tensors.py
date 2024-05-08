import json
import os

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as t
from tqdm import tqdm

from args import parse_save_dataset_tensors
from symile.cxr_prediction_age_sex.constants import GENDER_MAP


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


def save_dataset_tensors(args, df, split):
    save_dir = args.data_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cxr_list = []
    age_list = []
    gender_list = []
    dicom_id_list = []
    label_name_list = []
    label_value_list = []

    for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
        cxr = get_cxr(args, row["cxr_path"], split)
        cxr_list.append(cxr)

        age_list.append(int(row["age"]))

        gender_list.append(GENDER_MAP[row["gender"]])

        dicom_id_list.append(row["dicom_id"])

        if split in ["val_query", "val_candidate", "test_query", "test_candidate"]:
            label_name_list.append(row["label_name"])
            label_value_list.append(row["label_value"])

    cxr_tensor = torch.stack(cxr_list) # (n, 3, cxr_crop, cxr_crop)
    age_tensor = torch.tensor(age_list) # (n,)
    gender_tensor = torch.tensor(gender_list) # (n,)

    torch.save(cxr_tensor, save_dir / f"cxr_{split}.pt")
    torch.save(age_tensor, save_dir / f"age_{split}.pt")
    torch.save(gender_tensor, save_dir / f"gender_{split}.pt")

    with open(f"{save_dir}/dicom_id_{split}.txt", 'w') as f:
        f.writelines("\n".join(dicom_id_list))

    if split in ["val_query", "val_candidate", "test_query", "test_candidate"]:
        label_value_tensor = torch.tensor(label_value_list)
        torch.save(label_value_tensor, save_dir / f"label_value_{split}.pt")

        with open(f"{save_dir}/label_name_{split}.txt", 'w') as f:
            f.writelines("\n".join(label_name_list))


if __name__ == '__main__':
    args = parse_save_dataset_tensors()

    if args.split_to_run == "train":
        print("Saving train tensors...")
        train_df = pd.read_csv(args.data_dir / args.train_csv)
        save_dataset_tensors(args, train_df, "train")
    else:
        val_df = pd.read_csv(args.data_dir / args.val_csv)
        test_df = pd.read_csv(args.data_dir / args.test_csv)

        val_query_df = pd.read_csv(args.data_dir / args.val_query_csv)
        val_candidate_df = pd.read_csv(args.data_dir / args.val_candidate_csv)

        test_query_df = pd.read_csv(args.data_dir / args.test_query_csv)
        test_candidate_df = pd.read_csv(args.data_dir / args.test_candidate_csv)

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