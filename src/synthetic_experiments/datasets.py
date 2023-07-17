import numpy as np
from scipy.stats import uniform
import torch
from torch.utils.data import Dataset


def generate_data(n, eps_multiplier):
    """
    Generate n samples of data (A, B, C) for the modulo synthetic experiment.
    A, B, eps ~ Uniform(0,1), and C = ((A+B) % 1) + eps.

    Args:
        n (int): number of data samples to generate.
        eps_multiplier (float): multiplier for eps in C = ((A+B) % 1) + eps.
    Returns:
        A, B, C (tuple): each of A, B, C, is an numpy.ndarray of size (n,).
    """
    A = uniform.rvs(size=n)
    B = uniform.rvs(size=n)
    epsilon = uniform.rvs(size=n) * eps_multiplier
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


def get_vectors(A, B, C, i, d):
    """
    Create vectors (v_a, v_b, v_c) from the data samples (A, B, C).
    Each of A, B, C is a numpy.ndarray of size (n,).
    Each of v_a, v_b, v_c is a torch.Tensor of size (n, d).
    """
    v_a = torch.tensor([build_vector(a, i, d) for a in A], dtype=torch.float32)
    v_b = torch.tensor([build_vector(b, i, d) for b in B], dtype=torch.float32)
    v_c = torch.tensor([build_vector(c, i, d) for c in C], dtype=torch.float32)
    assert torch.all(v_a[:,i] == torch.tensor(A, dtype=torch.float32))
    assert torch.all(v_b[:,i] == torch.tensor(B, dtype=torch.float32))
    assert torch.all(v_c[:,i] == torch.tensor(C, dtype=torch.float32))
    return v_a, v_b, v_c


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
    def __init__(self, d, n, eps_multiplier, model):
        (self.A, self.B, self.C) = generate_data(n, eps_multiplier)
        i = np.random.randint(0, d)
        v_a, v_b, v_c = get_vectors(self.A, self.B, self.C, i, d)
        self.r_a, self.r_b, self.r_c, _ = self.get_representations(model, v_a, v_b, v_c)

        # binarize C
        C_thresh = 0.5 * (1.0 + eps_multiplier)
        self.C_bin = np.where(self.C <= C_thresh, 0, 1)

        assert self.r_a.shape == self.r_b.shape == self.r_c.shape, \
            "Vectors must be the same shape."
        assert self.r_a.shape[0] == self.C_bin.shape[0], \
            "Vectors and labels must be the same length."

    def get_representations(self, model, v_a, v_b, v_c):
        """
        Generate representations (r_a, r_b, r_c) from (v_a, v_b, v_c) using
        encoders in `model`.

        Args:
            model (nn.Module): model used to generate representations.
            v_a, v_b, v_c (torch.Tensor): each of size (n, d_v).
        Returns:
            r_a, r_b, r_c (torch.Tensor): each of size (n, d_r).
            logit_scale.exp() (torch.Tensor): temperature parameter
        """
        model.eval()
        with torch.no_grad():
            return model(v_a, v_b, v_c)

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
            Each instance of r_a, r_b, r_c is a torch.Tensor of size d_r.
            Each instance of A, B, C is a numpy.float64 scalar.
            C_bin (numpy.int64): label for the data sample, either 0 or 1.
        """
        return (self.A[idx], self.B[idx], self.C[idx],
                self.r_a[idx, :], self.r_b[idx, :], self.r_c[idx, :],
                self.C_bin[idx])


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
    def __init__(self, d, n, eps_multiplier):
        """
        Initialize the dataset object.

        Args:
            d (int): dimensionality for each of the vectors v_a, v_b, v_c.
            n (int): number of data samples to generate.
        """
        A, B, C = generate_data(n, eps_multiplier)
        i = np.random.randint(0, d)
        self.v_a, self.v_b, self.v_c = get_vectors(A, B, C, i, d)
        assert self.v_a.shape == self.v_b.shape == self.v_c.shape, \
            "All vectors must be the same shape."
        assert self.v_a.shape[1] == d, \
            "Vectors must have dimension d."

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