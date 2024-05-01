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
        ecg_train = torch.load(self.args.data_dir / "train/ecg_train.pt")
        labs_percentiles_train = torch.load(self.args.data_dir / "train/labs_percentiles_train.pt")
        labs_missingness_train = torch.load(self.args.data_dir / "train/labs_missingness_train.pt")
        risk_vector_train = torch.load(self.args.data_dir / "train/risk_vector_train.pt")
        hadm_id_train = torch.load(self.args.data_dir / "train/hadm_id_train.pt")

        ecg_val = torch.load(self.args.data_dir / "val/ecg_val.pt")
        labs_percentiles_val = torch.load(self.args.data_dir / "val/labs_percentiles_val.pt")
        labs_missingness_val = torch.load(self.args.data_dir / "val/labs_missingness_val.pt")
        risk_vector_val = torch.load(self.args.data_dir / "val/risk_vector_val.pt")
        hadm_id_val = torch.load(self.args.data_dir / "val/hadm_id_val.pt")

        ecg_test = torch.load(self.args.data_dir / "test/ecg_test.pt")
        labs_percentiles_test = torch.load(self.args.data_dir / "test/labs_percentiles_test.pt")
        labs_missingness_test = torch.load(self.args.data_dir / "test/labs_missingness_test.pt")
        risk_vector_test = torch.load(self.args.data_dir / "test/risk_vector_test.pt")
        hadm_id_test = torch.load(self.args.data_dir / "test/hadm_id_test.pt")

        self.ds_train = TensorDataset(ecg_train, labs_percentiles_train, labs_missingness_train,
                                      risk_vector_train, hadm_id_train)
        self.ds_val = TensorDataset(ecg_val, labs_percentiles_val, labs_missingness_val,
                                    risk_vector_val, hadm_id_val)
        self.ds_test = TensorDataset(ecg_test, labs_percentiles_test, labs_missingness_test,
                                     risk_vector_test, hadm_id_test)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)