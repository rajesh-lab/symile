import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class RiskPredictionDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        cxr_train = torch.load(self.args.data_dir / "train/cxr_train.pt")
        ecg_train = torch.load(self.args.data_dir / "train/ecg_train.pt")
        risk_vector_train = torch.load(self.args.data_dir / "train/risk_vector_train.pt")
        hadm_id_train = torch.load(self.args.data_dir / "train/hadm_id_train.pt")

        cxr_val = torch.load(self.args.data_dir / "val/cxr_val.pt")
        ecg_val = torch.load(self.args.data_dir / "val/ecg_val.pt")
        risk_vector_val = torch.load(self.args.data_dir / "val/risk_vector_val.pt")
        hadm_id_val = torch.load(self.args.data_dir / "val/hadm_id_val.pt")

        cxr_test = torch.load(self.args.data_dir / "test/cxr_test.pt")
        ecg_test = torch.load(self.args.data_dir / "test/ecg_test.pt")
        risk_vector_test = torch.load(self.args.data_dir / "test/risk_vector_test.pt")
        hadm_id_test = torch.load(self.args.data_dir / "test/hadm_id_test.pt")

        self.ds_train = TensorDataset(cxr_train, ecg_train, risk_vector_train,
                                      hadm_id_train)
        self.ds_val = TensorDataset(cxr_val, ecg_val, risk_vector_val,
                                    hadm_id_val)
        self.ds_test = TensorDataset(cxr_test, ecg_test, risk_vector_test,
                                     hadm_id_test)

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