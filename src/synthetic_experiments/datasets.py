import numpy as np
from scipy.stats import bernoulli
import torch
from torch.utils.data import Dataset


def generate_data_xor(d, n):
    """
    Generate n samples of data (v_a, v_b, v_c) for the XOR synthetic experiment.
    v_a[i], v_b[i] ~ Bernoulli(0.5), and v_c[i] = v_a[i] XOR v_b[i].

    Args:
        d (int): dimensionality of each of the vectors v_a, v_b, v_c.
        n (int): number of data samples to generate.
    Returns:
        v_a, v_b, v_c: each is an torch.Tensor of size (n,d).
    """
    p = 0.5
    v_a = bernoulli.rvs(p, size=(n,d))
    v_b = bernoulli.rvs(p, size=(n,d))
    v_c = np.bitwise_xor(v_a, v_b)
    
    v_a = torch.from_numpy(v_a).to(torch.float32)
    v_b = torch.from_numpy(v_b).to(torch.float32)
    v_c = torch.from_numpy(v_c).to(torch.float32)

    assert v_a.shape == v_b.shape == v_c.shape, \
        "Random variables must be the same shape"
    for arr in (v_a, v_b, v_c):
        assert torch.all((arr == 0) | (arr == 1)), "Random variables must be 0 or 1."
    assert v_a.shape[1] == d, "Vectors must have dimension d."
    return (v_a, v_b, v_c)


class PretrainingDataset(Dataset):
    """
    Pretraining dataset for the XOR synthetic experiment.
    Generate n samples of data (v_a, v_b, v_c) where
    v_a[i], v_b[i] ~ Bernoulli(0.5), and v_c[i] = v_a[i] XOR v_b[i].
    """
    def __init__(self, d, n):
        """
        Initialize the dataset object.

        Args:
            d (int): dimensionality for each of the vectors v_a, v_b, v_c.
            n (int): number of data samples to generate.
        """
        self.v_a, self.v_b, self.v_c = generate_data_xor(d, n)

    def __len__(self):
        """
        Compute length of the dataset.

        Args:
            n (int): dataset size.
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


def get_representations(model, v_a, v_b, v_c):
    """
    Generate representations (r_a, r_b, r_c) from (v_a, v_b, v_c) using
    encoders in `model`.

    Args:
        model (nn.Module): model used to generate representations.
        v_a, v_b, v_c (torch.Tensor): each of size (n, d_v).
    Returns:
        r_a, r_b, r_c (torch.Tensor): each of size (n, d_r).
    """
    model.eval()
    with torch.no_grad():
        r_a, r_b, r_c, _ = model(v_a, v_b, v_c)
        assert r_a.shape == r_b.shape == r_c.shape, "Vectors must be the same shape."
        return r_a, r_b, r_c


class ZeroshotTestDataset(Dataset):
    """
    Test dataset for the zeroshot classification evaluation.
    Generate n samples of data (v_a, v_b, v_c) where
    v_a[i], v_b[i] ~ Bernoulli(0.5), and v_c[i] = v_a[i] XOR v_b[i].
    We then generate the representations (r_a, r_b, r_c) from (v_a, v_b, v_c)
    using the provided encoders.
    """
    def __init__(self, d, n, model):
        """
        Initialize the dataset object.

        Args:
            d (int): dimensionality for each of the vectors v_a, v_b, v_c.
            n (int): number of data samples to generate.
        """
        v_a, v_b, v_c = generate_data_xor(d, n)
        self.r_a, self.r_b, self.r_c = get_representations(model, v_a, v_b, v_c)

    def __len__(self):
        """
        Compute length of the dataset.

        Args:
            n (int): dataset size.
        """
        return len(self.r_a)

    def __getitem__(self, idx):
        """
        Index into the dataset.

        Args:
            idx (int): index of data sample to retrieve.
        Returns
            Each returned instance of r_a, r_b, r_c is a torch.Tensor of size d_r.
        """
        return (self.r_a[idx, :], self.r_b[idx, :], self.r_c[idx, :])


def get_negative_v_c(v_c):
    """
    Samples a negative sample for each datapoint in v_c.
    """
    v_c_support = torch.Tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    v_c_neg = torch.clone(v_c)
    for r in range(len(v_c)):
        if torch.eq(v_c[r], torch.Tensor([0., 0.])).all():
            idx = np.random.choice([1, 2, 3])
            v_c_neg[r] = v_c_support[idx]
        elif torch.eq(v_c[r], torch.Tensor([0., 1.])).all():
            idx = np.random.choice([0, 2, 3])
            v_c_neg[r] = v_c_support[idx]
        elif torch.eq(v_c[r], torch.Tensor([1., 0.])).all():
            idx = np.random.choice([0, 1, 3])
            v_c_neg[r] = v_c_support[idx]
        elif torch.eq(v_c[r], torch.Tensor([1., 1.])).all():
            idx = np.random.choice([0, 1, 2])
            v_c_neg[r] = v_c_support[idx]
    assert torch.eq(v_c_neg, v_c).all() == False, "There can be no false negative samples."
    return v_c_neg


class SupportTestDataset(Dataset):
    """
    Test dataset for the in support classification evaluation with
    n // 2 samples of positive triples (v_a, v_b, v_c) where
    v_a[i], v_b[i] ~ Bernoulli(0.5), and v_c[i] = v_a[i] XOR v_b[i].
    and n // 2 negative triples.
    Representations (r_a, r_b, r_c) are generated from both positive and
    negative triples (v_a, v_b, v_c) using the provided encoders.
    """
    def __init__(self, d, n, model):
        """
        Initialize the dataset object.

        Args:
            d (int): dimensionality for each of the vectors v_a, v_b, v_c.
            n (int): number of data samples to generate.
        """
        pos_n = n // 2
        v_a, v_b, v_c = generate_data_xor(d, pos_n)
        r_a, r_b, r_c = get_representations(model, v_a, v_b, v_c)

        # get v_c's for out of support triples
        v_c_neg = get_negative_v_c(v_c)
        r_a_neg, r_b_neg, r_c_neg = get_representations(model, v_a, v_b, v_c_neg)
        assert torch.eq(r_a, r_a_neg).all() and torch.eq(r_b, r_b_neg).all(), \
            "r_a and r_b should be the same for positive and negative triples."

        self.r_a = torch.concat((r_a, r_a), axis=0)
        self.r_b = torch.concat((r_b, r_b), axis=0)
        self.r_c = torch.concat((r_c, r_c_neg), axis=0)
        self.in_support = torch.concat((torch.ones(r_a.shape[0]),
                                        torch.zeros(r_a.shape[0])), axis=0)

    def __len__(self):
        """
        Compute length of the dataset.

        Args:
            n (int): dataset size.
        """
        return len(self.r_a)

    def __getitem__(self, idx):
        """
        Index into the dataset.

        Args:
            idx (int): index of data sample to retrieve.
        Returns
            Each returned instance of r_a, r_b, r_c is a torch.Tensor of size d_r.
            self.in_support[idx] (torch.Tensor): label for the data sample,
                                                 either 0.0 or 1.0 (float).
        """
        return (self.r_a[idx, :], self.r_b[idx, :], self.r_c[idx, :],
                self.in_support[idx])


class SumTestDataset(Dataset):
    """
    Test dataset for the representation sum classification evaluation.
    Generate n samples of data (v_a, v_b, v_c) where
    v_a[i], v_b[i] ~ Bernoulli(0.5), and v_c[i] = v_a[i] XOR v_b[i].
    We then generate the representations (r_a, r_b) from (v_a, v_b)
    using the provided encoders.
    The label for each sample is the sum of the elements of v_c.
    """
    def __init__(self, d, n, model):
        """
        Initialize the dataset object.

        Args:
            d (int): dimensionality for each of the vectors v_a, v_b, v_c.
            n (int): number of data samples to generate.
        """
        v_a, v_b, v_c = generate_data_xor(d, n)
        self.r_a, self.r_b, _ = get_representations(model, v_a, v_b, v_c)
        self.sum = v_c.sum(axis=1)

    def __len__(self):
        """
        Compute length of the dataset.

        Args:
            n (int): dataset size.
        """
        return len(self.r_a)

    def __getitem__(self, idx):
        """
        Index into the dataset.

        Args:
            idx (int): index of data sample to retrieve.
        Returns
            Each instance of r_a, r_b is a torch.Tensor of size d_r.
            self.sum[idx] (float): label for the data sample (0.0, 1.0, or 2.0).
        """
        return (self.r_a[idx, :], self.r_b[idx, :], self.sum[idx])