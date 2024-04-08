import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


from src.healthcare.constants import *


class EvaluationDataset(Dataset):
    def __init__(self, args, split, type):
        split_suffix = "" if split == "test" else "_val"

        self.cxr = torch.load(args.data_dir / f"cxr_{type}{split_suffix}.pt")
        self.ecg = torch.load(args.data_dir / f"ecg_{type}{split_suffix}.pt")
        self.labs_percentiles = torch.load(args.data_dir / f"labs_percentiles_{type}{split_suffix}.pt")
        self.labs_missingness = torch.load(args.data_dir / f"labs_missingness_{type}{split_suffix}.pt")
        self.hadm_id = torch.load(args.data_dir / f"hadm_id_{type}{split_suffix}.pt")
        self.label_value = torch.load(args.data_dir / f"label_value_{type}{split_suffix}.pt")

        with open(f"{args.data_dir}/label_name_{type}{split_suffix}.txt", 'r') as f:
            self.label_name = f.read().splitlines()

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        return {"cxr": self.cxr[idx],
                "ecg": self.ecg[idx],
                "labs_percentiles": self.labs_percentiles[idx],
                "labs_missingness": self.labs_missingness[idx],
                "hadm_id": self.hadm_id[idx],
                "label_name": self.label_name[idx],
                "label_value": self.label_value[idx]}


class HighDimDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        cxr_train = torch.load(self.args.data_dir / "cxr_train.pt")
        ecg_train = torch.load(self.args.data_dir / "ecg_train.pt")
        labs_percentiles_train = torch.load(self.args.data_dir / "labs_percentiles_train.pt")
        labs_missingness_train = torch.load(self.args.data_dir / "labs_missingness_train.pt")

        cxr_val = torch.load(self.args.data_dir / "cxr_val.pt")
        ecg_val = torch.load(self.args.data_dir / "ecg_val.pt")
        labs_percentiles_val = torch.load(self.args.data_dir / "labs_percentiles_val.pt")
        labs_missingness_val = torch.load(self.args.data_dir / "labs_missingness_val.pt")

        self.ds_train = TensorDataset(cxr_train, ecg_train,
                                      labs_percentiles_train, labs_missingness_train)
        self.ds_val = TensorDataset(cxr_val, ecg_val,
                                    labs_percentiles_val, labs_missingness_val)

        self.ds_query_test = EvaluationDataset(self.args, "test", "query")

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=len(self.ds_val),
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_query_test, batch_size=len(self.ds_query_test),
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)