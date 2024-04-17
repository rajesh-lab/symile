import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class EvaluationDataset(Dataset):
    def __init__(self, args, split, type):
        self.cxr = torch.load(args.data_dir / f"{split}_{type}/cxr_{split}_{type}.pt")
        self.ecg = torch.load(args.data_dir / f"{split}_{type}/ecg_{split}_{type}.pt")
        self.labs_percentiles = torch.load(args.data_dir / f"{split}_{type}/labs_percentiles_{split}_{type}.pt")
        self.labs_missingness = torch.load(args.data_dir / f"{split}_{type}/labs_missingness_{split}_{type}.pt")
        self.hadm_id = torch.load(args.data_dir / f"{split}_{type}/hadm_id_{split}_{type}.pt")
        self.label_value = torch.load(args.data_dir / f"{split}_{type}/label_value_{split}_{type}.pt")

        with open(f"{args.data_dir}/{split}_{type}/label_name_{split}_{type}.txt", 'r') as f:
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
        cxr_train = torch.load(self.args.data_dir / "train/cxr_train.pt")
        ecg_train = torch.load(self.args.data_dir / "train/ecg_train.pt")
        labs_percentiles_train = torch.load(self.args.data_dir / "train/labs_percentiles_train.pt")
        labs_missingness_train = torch.load(self.args.data_dir / "train/labs_missingness_train.pt")
        hadm_id_train = torch.load(self.args.data_dir / "train/hadm_id_train.pt")

        cxr_val = torch.load(self.args.data_dir / "val/cxr_val.pt")
        ecg_val = torch.load(self.args.data_dir / "val/ecg_val.pt")
        labs_percentiles_val = torch.load(self.args.data_dir / "val/labs_percentiles_val.pt")
        labs_missingness_val = torch.load(self.args.data_dir / "val/labs_missingness_val.pt")
        hadm_id_val = torch.load(self.args.data_dir / "val/hadm_id_val.pt")

        cxr_test = torch.load(self.args.data_dir / "test/cxr_test.pt")
        ecg_test = torch.load(self.args.data_dir / "test/ecg_test.pt")
        labs_percentiles_test = torch.load(self.args.data_dir / "test/labs_percentiles_test.pt")
        labs_missingness_test = torch.load(self.args.data_dir / "test/labs_missingness_test.pt")
        hadm_id_test = torch.load(self.args.data_dir / "test/hadm_id_test.pt")

        assert torch.unique(hadm_id_train).numel() == hadm_id_train.numel()
        assert torch.unique(hadm_id_val).numel() == hadm_id_val.numel()
        assert torch.unique(hadm_id_test).numel() == hadm_id_test.numel()

        self.ds_train = TensorDataset(cxr_train, ecg_train, labs_percentiles_train,
                                      labs_missingness_train, hadm_id_train)
        self.ds_val = TensorDataset(cxr_val, ecg_val, labs_percentiles_val,
                                    labs_missingness_val, hadm_id_val)
        self.ds_test = TensorDataset(cxr_test, ecg_test, labs_percentiles_test,
                                     labs_missingness_test, hadm_id_test)

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