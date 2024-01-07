import os

import numpy as np
import lightning.pytorch as pl
from scipy.stats import bernoulli
import torch
from torch.utils.data import Dataset, DataLoader


class BinaryDataset(Dataset):
    """
    Generate n samples of data (v_a, v_b, v_c) according to the below.

    i ~ Bernoulli(i_p)
    dim(v_a) = dim(v_b) = dim(v_c) = d
    v_a[j], v_b[j] ~ Bernoulli(0.5)
    v_c[j] = (v_a[j] XOR v_b[j])^i * v_a[j]^(1-i)
    """
    def __init__(self, d, n, i_p):
        """
        Args:
            d (int): dimensionality for each of the vectors v_a, v_b, v_c.
            n (int): number of data samples to generate.
            i_p (float): Bernoulli distribution parameter for i.
        """
        self.v_a, self.v_b, self.v_c = self.generate_data(d, n, i_p)

    def generate_data(self, d, n, i_p):
        """
        Returns:
            v_a, v_b, v_c: each is an torch.Tensor of size (n, d).
        """
        v_a = bernoulli.rvs(0.5, size=(n, d))
        v_b = bernoulli.rvs(0.5, size=(n, d))
        i = bernoulli.rvs(i_p, size=n)

        xor = np.bitwise_xor(v_a, v_b)

        if d == 1:
            i = np.expand_dims(i, axis=1)
            v_c = np.where(i, xor, v_a)
        else: # d > 1
            c_columns = []
            for j in range(d):
                c_columns.append(np.where(i, xor[:, j], v_a[:, j]))
            v_c = np.stack(c_columns, axis=1)

        v_a = torch.from_numpy(v_a).to(torch.float32)
        v_b = torch.from_numpy(v_b).to(torch.float32)
        v_c = torch.from_numpy(v_c).to(torch.float32)

        assert v_a.shape == v_b.shape == v_c.shape, \
            "Random variables must be the same shape"
        for arr in (v_a, v_b, v_c):
            assert torch.all((arr == 0) | (arr == 1)), "Random variables must be 0 or 1."
        assert v_a.shape[1] == d, "Vectors must have dimension d."

        return v_a, v_b, v_c

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
            v_a, v_b, v_c (tuple): each of v_a, v_b, v_c is a torch.Tensor of size d.
        """
        v_a = self.v_a[idx, :]
        v_b = self.v_b[idx, :]
        v_c = self.v_c[idx, :]
        return v_a, v_b, v_c


class BinaryDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        self.ds_train = BinaryDataset(self.args.d_v, self.args.pretrain_n,
                                      self.args.i_p)
        self.ds_val = BinaryDataset(self.args.d_v, self.args.pretrain_val_n,
                                    self.args.i_p)
        self.ds_test = BinaryDataset(self.args.d_v, self.args.test_n,
                                     self.args.i_p)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train, shuffle=True,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.pretrain_val_n,
                          num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          num_workers=self.num_workers, drop_last=True)