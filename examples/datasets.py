import os

import numpy as np
import lightning.pytorch as pl
from scipy.stats import bernoulli
import torch
from torch.utils.data import Dataset, DataLoader


class BinaryXOR8Dataset(Dataset):
    """
    Generate n samples of data (v_a, v_b, ..., v_h) where
    v_a, v_b, ..., v_g ~ Bernoulli(0.5)
    v_h = v_a XOR v_b XOR v_c XOR ... XOR v_g
    """
    def __init__(self, n):
        """
        Args:
            n (int): number of data samples to generate.
        """
        self.v_a, self.v_b, self.v_c, self.v_d, self.v_e, self.v_f, self.v_g, self.v_h = self.generate_data(n)

    def generate_data(self, n):
        """
        Returns:
            v_a, v_b, ..., v_h: each is a torch.Tensor of size (n).
        """
        v_a = bernoulli.rvs(0.5, size=n)
        v_b = bernoulli.rvs(0.5, size=n)
        v_c = bernoulli.rvs(0.5, size=n)
        v_d = bernoulli.rvs(0.5, size=n)
        v_e = bernoulli.rvs(0.5, size=n)
        v_f = bernoulli.rvs(0.5, size=n)
        v_g = bernoulli.rvs(0.5, size=n)
        v_h = np.bitwise_xor.reduce([v_a, v_b, v_c, v_d, v_e, v_f, v_g])

        v_a = torch.from_numpy(v_a).float().unsqueeze(1)
        v_b = torch.from_numpy(v_b).float().unsqueeze(1)
        v_c = torch.from_numpy(v_c).float().unsqueeze(1)
        v_d = torch.from_numpy(v_d).float().unsqueeze(1)
        v_e = torch.from_numpy(v_e).float().unsqueeze(1)
        v_f = torch.from_numpy(v_f).float().unsqueeze(1)
        v_g = torch.from_numpy(v_g).float().unsqueeze(1)
        v_h = torch.from_numpy(v_h).float().unsqueeze(1)

        return v_a, v_b, v_c, v_d, v_e, v_f, v_g, v_h

    def __len__(self):
        """
        Compute length of the dataset.

        Returns:
            (int): dataset size.
        """
        return len(self.v_a)

    def __getitem__(self, idx):
        """
        Index into the dataset.

        Args:
            idx (int): index of data sample to retrieve.
        Returns
            v_a, v_b, ..., v_j (tuple): each of v_a, v_b, ..., v_j is a torch.Tensor.
        """
        return (self.v_a[idx], self.v_b[idx], self.v_c[idx], self.v_d[idx],
                self.v_e[idx], self.v_f[idx], self.v_g[idx], self.v_h[idx])


class BinaryXOR8DataModule(pl.LightningDataModule):
    def __init__(self, args):
        """
        Initialize LightningDataModule for the binary XOR 10 dataset.

        Args:
            args (Namespace): contains arguments for dataset configuration and training.
        """
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        self.ds_train = BinaryXOR8Dataset(self.args.train_n)
        self.ds_val = BinaryXOR8Dataset(self.args.val_n)
        self.ds_test = BinaryXOR8Dataset(self.args.test_n)

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

    def resample_test_set(self):
        self.ds_test = BinaryXOR8Dataset(self.args.test_n)