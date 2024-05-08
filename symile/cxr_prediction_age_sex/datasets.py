import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as t

from symile.cxr_prediction_age_sex.constants import IMAGENET_MEAN, IMAGENET_STD


class CXRPredictionDataset(Dataset):
    def __init__(self, args, split, type=None):
        self.args = args

        self.type = type
        split_type = f"{split}_{type}" if self.type else split

        self.split_dir = self.args.data_dir / f"{split_type}"

        self.cxr = torch.load(self.split_dir / f"cxr_{split_type}.pt")
        self.age = torch.load(self.split_dir / f"age_{split_type}.pt")
        self.gender = torch.load(self.split_dir / f"gender_{split_type}.pt")

        with open(f"{self.split_dir}/dicom_id_{split_type}.txt", 'r') as f:
            self.dicom_id = f.read().splitlines()

        if self.type:
            self.label_value = torch.load(f"{self.split_dir}/label_value_{split_type}.pt")

            with open(f"{self.split_dir}/label_name_{split_type}.txt", 'r') as f:
                self.label_name = f.read().splitlines()

    def __len__(self):
        return len(self.cxr)

    def normalize_cxr(self, cxr):
        return t.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(cxr)

    def get_cxr(self, idx):
        cxr = self.cxr[idx]
        cxr = cxr.float() / 255
        cxr = self.normalize_cxr(cxr)
        return cxr # (3, H, W)

    def __getitem__(self, idx):
        data = {"cxr": self.get_cxr(idx),
                "age": self.age[idx],
                "gender": self.gender[idx],
                "dicom_id": self.dicom_id[idx]}

        if self.type:
            data["label_value"] = self.label_value[idx]
            data["label_name"] = self.label_name[idx]

        return data


class CXRPredictionDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        self.ds_train = CXRPredictionDataset(self.args, "train")
        self.ds_val = CXRPredictionDataset(self.args, "val")

        # Test phase is not processed in batches, but in order for Lightning to execute
        # test phase, a dummy test_dataloader() needs to be provided.
        self.ds_test = TensorDataset(torch.zeros(1))

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)