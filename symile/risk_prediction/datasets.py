import json
import os

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as t

from symile.risk_prediction.constants import *


class RiskPredictionDataset(Dataset):
    def __init__(self, args, split):
        self.args = args

        self.split_dir = self.args.data_dir / f"{split}"

        self.df = pd.read_csv(self.args.data_dir / f"{split}.csv")

        self.cxr = torch.load(self.split_dir / f"cxr_{split}.pt")
        self.ecg = torch.load(self.split_dir / f"ecg_{split}.pt")
        self.risk_vector = torch.load(self.split_dir / f"risk_vector_{split}.pt")
        self.hadm_id = torch.load(self.split_dir / f"hadm_id_{split}.pt")

        with open(self.split_dir / f"cxr_mapping_{split}.json", "r") as f:
            self.cxr_mapping = json.load(f)
        with open(self.split_dir / f"ecg_mapping_{split}.json", "r") as f:
            self.ecg_mapping = json.load(f)

        mean_cxr = torch.load(self.args.mean_cxr)
        self.mean_cxr = self.normalize_cxr(mean_cxr)
        _, self.cxr_height, self.cxr_width = self.mean_cxr.shape

        self.mean_ecg = torch.load(self.args.mean_ecg)
        _, self.ecg_length, self.ecg_channels = self.mean_ecg.shape

    def __len__(self):
        return len(self.df)

    def normalize_cxr(self, cxr):
        return t.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(cxr)

    def process_cxr(self, cxr):
        cxr = cxr.float() / 255
        return self.normalize_cxr(cxr)

    def get_cxr(self, row, cxr_is_missing):
        if cxr_is_missing:
            cxr = torch.cat((torch.zeros(1, self.cxr_height, self.cxr_width), self.mean_cxr), 0)
        else:
            cxr_idx = self.cxr_mapping[str(row["hadm_id"])]
            cxr = self.cxr[cxr_idx] # (3, H, W)
            cxr = self.process_cxr(cxr)
            cxr = torch.cat((torch.ones(1, self.cxr_height, self.cxr_width), cxr), 0)
        return cxr # (4, H, W)

    def get_ecg(self, row, ecg_is_missing):
        if ecg_is_missing:
            ecg = torch.cat((torch.zeros(1, self.ecg_length, self.ecg_channels), self.mean_ecg), 0)
        else:
            ecg_idx = self.ecg_mapping[str(row["hadm_id"])]
            ecg = self.ecg[ecg_idx] # (1, 5000, 12)
            ecg = torch.cat((torch.ones(1, self.ecg_length, self.ecg_channels), ecg), 0)
        return ecg # (2, 5000, 12)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cxr_is_missing = False if pd.notnull(row["cxr_path"]) else True
        cxr = self.get_cxr(row, cxr_is_missing)

        ecg_is_missing = False if pd.notnull(row["ecg_path"]) else True
        ecg = self.get_ecg(row, ecg_is_missing)

        risk_vector = torch.tensor(row[RISK_VECTOR_COLS], dtype=torch.float32)

        return {"cxr": cxr, "cxr_missing": cxr_is_missing,
                "ecg": ecg, "ecg_is_missing": ecg_is_missing,
                "risk_vector": risk_vector,
                "hadm_id": row["hadm_id"]}


class RiskPredictionDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        self.ds_train = RiskPredictionDataset(self.args, "train")
        self.ds_val = RiskPredictionDataset(self.args, "val")
        self.ds_test = RiskPredictionDataset(self.args, "test")

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