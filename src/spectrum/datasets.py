import os

import numpy as np
import lightning.pytorch as pl
from scipy.stats import bernoulli
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    # """
    # Dataset for the XOR synthetic experiment.
    # Generate n samples of data (v_a, v_b, v_c) where
    # v_a[i], v_b[i] ~ Bernoulli(0.5), and v_c[i] = v_a[i] XOR v_b[i].
    # """
    def __init__(self, d, n, i_p=0.0):
        # """
        # Initialize the dataset object.

        # Args:
        #     d (int): dimensionality for each of the vectors v_a, v_b, v_c.
        #     n (int): number of data samples to generate.
        # """
        self.v_a, self.v_b, self.v_c = self.generate_data(d, n, i_p)

    def generate_data(self, d, n, i_p):
        # """
        # Generate n samples of data (v_a, v_b, v_c) for the XOR synthetic experiment.
        # v_a[i], v_b[i] ~ Bernoulli(0.5), and v_c[i] = v_a[i] XOR v_b[i].

        # Args:
        #     d (int): dimensionality of each of the vectors v_a, v_b, v_c.
        #     n (int): number of data samples to generate.
        # Returns:
        #     v_a, v_b, v_c: each is an torch.Tensor of size (n,d).
        # """
        v_a = bernoulli.rvs(0.5, size=(n, d))
        v_b = bernoulli.rvs(0.5, size=(n, d))
        i = bernoulli.rvs(i_p, size=n)

        xor = np.bitwise_xor(v_a, v_b)

        if d == 1:
            i = np.expand_dims(i, axis=1)
            v_c = np.where(i, xor, v_a)
        elif d == 2:
            v_c_0 = np.where(i, xor[:, 0], v_a[:, 0])
            v_c_1 = np.where(i, xor[:, 1], v_b[:, 0])
            v_c = np.stack([v_c_0, v_c_1], axis=1)
        else:
            raise ValueError("d must be 1 or 2.")

        v_a = torch.from_numpy(v_a).to(torch.float32)
        v_b = torch.from_numpy(v_b).to(torch.float32)
        v_c = torch.from_numpy(v_c).to(torch.float32)

        assert v_a.shape == v_b.shape == v_c.shape, \
            "Random variables must be the same shape"
        for arr in (v_a, v_b, v_c):
            assert torch.all((arr == 0) | (arr == 1)), "Random variables must be 0 or 1."
        assert v_a.shape[1] == d, "Vectors must have dimension d."

        return (v_a, v_b, v_c)

    def __len__(self):
        # """
        # Compute length of the dataset.

        # Returns:
        #     (int): dataset size.
        # """
        return len(self.v_a)

    def __getitem__(self, idx):
        # """
        # Index into the dataset.

        # Args:
        #     idx (int): index of data sample to retrieve.
        # Returns
        #     v_a, v_b, v_c (tuple): each of v_a, v_b, v_c is a torch.Tensor of size d.
        # """
        v_a = self.v_a[idx, :]
        v_b = self.v_b[idx, :]
        v_c = self.v_c[idx, :]
        return v_a, v_b, v_c


# class SyntheticSupportDataset(SyntheticDataset):
#     """
#     Test dataset for the in support classification evaluation with
#     n // 2 samples of positive triples (v_a, v_b, v_c) where
#     v_a[i], v_b[i] ~ Bernoulli(0.5), and v_c[i] = v_a[i] XOR v_b[i].
#     and n // 2 negative triples.
#     Representations (r_a, r_b, r_c) are generated from both positive and
#     negative triples (v_a, v_b, v_c) using the provided encoders.
#     """
#     def __init__(self, d, n, p=0.0):
#         super().__init__(d, n, p=0.0)
#         pos_n = n // 2
#         v_a, v_b, v_c = self.generate_data_xor(d, pos_n)

#         # get v_c's for out of support triples
#         v_c_neg = self.get_negative_v_c(v_c)

#         self.v_a = torch.concat((v_a, v_a), axis=0)
#         self.v_b = torch.concat((v_b, v_b), axis=0)
#         self.v_c = torch.concat((v_c, v_c_neg), axis=0)

#         self.in_support = torch.concat((torch.ones(v_a.shape[0]),
#                                         torch.zeros(v_a.shape[0])), axis=0)

#         self.v_a_missingness = self.assign_missingness(n, p)
#         self.v_b_missingness = self.assign_missingness(n, p)
#         self.v_c_missingness = self.assign_missingness(n, p)

#     def get_negative_v_c(self, v_c):
#         """
#         Samples a negative sample for each datapoint in v_c.
#         """
#         v_c_support = torch.Tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
#         v_c_neg = torch.clone(v_c)

#         for r in range(len(v_c)):
#             if torch.eq(v_c[r], torch.Tensor([0., 0.])).all():
#                 idx = np.random.choice([1, 2, 3])
#                 v_c_neg[r] = v_c_support[idx]
#             elif torch.eq(v_c[r], torch.Tensor([0., 1.])).all():
#                 idx = np.random.choice([0, 2, 3])
#                 v_c_neg[r] = v_c_support[idx]
#             elif torch.eq(v_c[r], torch.Tensor([1., 0.])).all():
#                 idx = np.random.choice([0, 1, 3])
#                 v_c_neg[r] = v_c_support[idx]
#             elif torch.eq(v_c[r], torch.Tensor([1., 1.])).all():
#                 idx = np.random.choice([0, 1, 2])
#                 v_c_neg[r] = v_c_support[idx]

#         assert torch.eq(v_c_neg, v_c).all() == False, "There can be no false negative samples."

#         return v_c_neg

#     def __len__(self):
#         """
#         Compute length of the dataset.

#         Returns:
#             (int): dataset size.
#         """
#         return len(self.v_a)

#     def __getitem__(self, idx):
#         """
#         Index into the dataset.

#         Args:
#             idx (int): index of data sample to retrieve.
#         Returns
#             Each returned instance of r_a, r_b, r_c is a torch.Tensor of size d_r.
#             self.in_support[idx] (torch.Tensor): label for the data sample,
#                                                  either 0.0 or 1.0 (float).
#         """
#         return (self.v_a[idx, :], self.v_b[idx, :], self.v_c[idx, :],
#                 self.v_a_missingness[idx],
#                 self.v_b_missingness[idx],
#                 self.v_c_missingness[idx],
#                 self.in_support[idx])


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        self.ds_train = SyntheticDataset(self.args.d_v, self.args.pretrain_n,
                                         self.args.i_p)
        self.ds_val = SyntheticDataset(self.args.d_v, self.args.pretrain_val_n,
                                       self.args.i_p)

        if self.args.evaluation == "zeroshot":
            self.ds_test = SyntheticDataset(self.args.d_v, self.args.test_n,
                                            self.args.i_p)
        # elif self.args.evaluation == "support":
        #     self.ds_test = XORSupportDataset(self.args.d_v, self.args.test_n,
        #                                      self.args.i_p)

    def train_dataloader(self):
        batch_sz = self.args.pretrain_n if self.args.use_full_dataset else self.args.batch_sz
        return DataLoader(self.ds_train, batch_size=batch_sz, shuffle=True,
                          num_workers=self.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.pretrain_val_n,
                          num_workers=self.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.test_n,
                          num_workers=self.num_workers, drop_last=True)