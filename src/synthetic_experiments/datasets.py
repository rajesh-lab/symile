import numpy as np
from scipy.stats import uniform
import torch
from torch.utils.data import Dataset


def generate_data(n):
    """
    Generate n samples of data (A, B, C) for the modulo synthetic experiment.
    A, B, eps ~ Uniform(0,1), and C = ((A+B) % 1) + eps.

    Args:
        n (int): number of data samples to generate.
    Returns:
        A, B, C (tuple): each of A, B, C, is an numpy.ndarray of size (n,).
    """
    A = uniform.rvs(size=n)
    B = uniform.rvs(size=n)
    epsilon = uniform.rvs(size=n)
    assert A.shape == B.shape == epsilon.shape, \
        "Random variables must be the same shape"
    for arr in (A, B, epsilon):
        assert np.all(arr >= 0) and np.all(arr <= 1), \
            "Random variables must be in [0,1]"

    def calculate_c(a, b):
        c = a + b - 1 if a + b > 1 else a + b
        return c
    C = np.vectorize(calculate_c)(A, B)
    C += epsilon

    def test_calculate_c(A, B, C, epsilon):
        for i in [0, -1]:
            a = A[i]
            b = B[i]
            c = C[i]
            e = epsilon[i]
            if a + b > 1:
                assert c == a + b - 1 + e
            else:
                assert c == a + b + e
    test_calculate_c(A, B, C, epsilon)

    return (A, B, C)


def build_vector(x, i, d):
    v_x = uniform.rvs(size=d)
    v_x[i] = x
    return v_x


class FinetuningDataset(Dataset):
    """
    Finetuning dataset for the modulo synthetic experiment.
    Generates n samples of (A, B, C) where:
        -- A, B, eps ~ Uniform(0,1)
        -- C = ((A+B) % 1) + eps
        -- C_bin = 0 if C <= 1.0 else 1
        -- i ~ Uniform(1, d)
    For each sample (A, B, C), we create vectors (v_a, v_b), each of size d_v,
    where v_a[i] = A, v_b[i] = B, and v_a[j!=i], v_b[j!=i] ~ Uniform(0,1).
    We then generate the representations (r_a, r_b) from (v_a, v_b) using the
    provided encoders.
    """
    def __init__(self, d, n, model):
        (A, B, C) = generate_data(n)
        i = np.random.randint(0, d)
        v_a, v_b = self.get_vectors(A, B, i, d)
        self.r_a, self.r_b, _, _ = self.get_representations(model, v_a, v_b)
        # binarize C
        self.C_bin = np.where(C <= 1.0, 0, 1)
        assert self.r_a.shape == self.r_b.shape, \
            "Vectors must be the same shape."
        assert self.r_a.shape[0] == self.C_bin.shape[0], \
            "Vectors and labels must be the same length."

    def get_vectors(self, A, B, i, d):
        """
        Create vectors (v_a, v_b) from the data samples (A, B).
        Each of A, B is a numpy.ndarray of size (n,).
        Each of v_a, v_b is a torch.Tensor of size (n, d_v).
        """
        v_a = torch.tensor([build_vector(a, i, d) for a in A], dtype=torch.float32)
        v_b = torch.tensor([build_vector(b, i, d) for b in B], dtype=torch.float32)
        assert torch.all(v_a[:,i] == torch.tensor(A, dtype=torch.float32))
        assert torch.all(v_b[:,i] == torch.tensor(B, dtype=torch.float32))
        return v_a, v_b

    def get_representations(self, model, v_a, v_b):
        """
        Generate representations (r_a, r_b) from (v_a, v_b) using encoders in
        `model`.

        Args:
            model (nn.Module): model used to generate representations.
            v_a (torch.Tensor): vector of size (n, d_v).
            v_b (torch.Tensor): vector of size (n, d_v).
        Returns:
            r_a, r_b (tuple): each of r_a, r_b is a torch.Tensor of size (n, d_r).
        """
        model.eval()
        with torch.no_grad():
            return model(v_a, v_b)

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
            r_a, r_b: each of r_a, r_b is a torch.Tensor of size d_r.
            C_bin (numpy.int64): label for the data sample, either 0 or 1.
        """
        r_a = self.r_a[idx, :]
        r_b = self.r_b[idx, :]
        C_bin = self.C_bin[idx]
        return r_a, r_b, C_bin


class PretrainingDataset(Dataset):
    """
    Pretraining dataset for the modulo synthetic experiment.
    Generates n samples of (A, B, C) where:
        -- A, B, eps ~ Uniform(0,1)
        -- C = ((A+B) % 1) + eps
        -- i ~ Uniform(1, d)
    For each sample (A, B, C), we create vectors (v_a, v_b, v_c), each of size d,
    where v_a[i] = A, v_b[i] = B, v_c[i] = C, and v_a[j!=i], v_b[j!=i], v_c[j!=i] ~ Uniform(0,1).
    """
    def __init__(self, d, n):
        """
        Initialize the dataset object.

        Args:
            d (int): dimensionality for each of the vectors v_a, v_b, v_c.
            n (int): number of data samples to generate.
        """
        data = generate_data(n)
        i = np.random.randint(0, d)
        self.v_a, self.v_b, self.v_c = self.get_vectors(data, i, d)
        assert self.v_a.shape == self.v_b.shape == self.v_c.shape, \
            "All vectors must be the same shape."
        assert self.v_a.shape[1] == d, \
            "Vectors must have dimension d."

    def get_vectors(self, data, i, d):
        """
        Create vectors (v_a, v_b, v_c) from the data samples (A, B, C).
        Each of A, B, C is a numpy.ndarray of size (n,).
        Each of v_a, v_b, v_c is a torch.Tensor of size (n, d).
        """
        A, B, C = data
        v_a = torch.tensor([build_vector(a, i, d) for a in A], dtype=torch.float32)
        v_b = torch.tensor([build_vector(b, i, d) for b in B], dtype=torch.float32)
        v_c = torch.tensor([build_vector(c, i, d) for c in C], dtype=torch.float32)
        assert torch.all(v_a[:,i] == torch.tensor(A, dtype=torch.float32))
        assert torch.all(v_b[:,i] == torch.tensor(B, dtype=torch.float32))
        assert torch.all(v_c[:,i] == torch.tensor(C, dtype=torch.float32))
        return v_a, v_b, v_c

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