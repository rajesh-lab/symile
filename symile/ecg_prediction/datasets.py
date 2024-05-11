import os

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, TensorDataset


class ECGPredictionDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        ecg_train = torch.load(self.args.data_dir / "train/ecg_train.pt")
        age_train = torch.load(self.args.data_dir / "train/age_train.pt")
        gender_train = torch.load(self.args.data_dir / "train/gender_train.pt")
        study_id_train = torch.load(self.args.data_dir / "train/study_id_train.pt")

        ecg_val = torch.load(self.args.data_dir / "val/ecg_val.pt")
        age_val = torch.load(self.args.data_dir / "val/age_val.pt")
        gender_val = torch.load(self.args.data_dir / "val/gender_val.pt")
        study_id_val = torch.load(self.args.data_dir / "val/study_id_val.pt")

        ecg_test = torch.load(self.args.data_dir / "test/ecg_test.pt")
        age_test = torch.load(self.args.data_dir / "test/age_test.pt")
        gender_test = torch.load(self.args.data_dir / "test/gender_test.pt")
        study_id_test = torch.load(self.args.data_dir / "test/study_id_test.pt")

        self.ds_train = TensorDataset(ecg_train, age_train, gender_train, study_id_train)
        self.ds_val = TensorDataset(ecg_val, age_val, gender_val, study_id_val)
        self.ds_val = TensorDataset(ecg_test, age_test, gender_test, study_id_test)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=True)