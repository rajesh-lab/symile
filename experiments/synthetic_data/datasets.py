import numpy as np
from scipy.stats import uniform
import torch
from torch.utils.data import Dataset

class FinetuningDataset(Dataset):
    def __init__(self, d, n, model):
            (A, B, self.C) = self.generate_data(n)
            i = np.random.randint(0, d)
            v_a, v_b = self.get_vectors(A, B, i, d)
            self.r_a, self.r_b = self.get_representations(model, v_a, v_b)
            assert self.r_a.shape == self.r_b.shape, \
                "Vectors must be the same shape."
            assert self.r_a.shape[0] == self.C.shape[0], \
                "Vectors and labels must be the same length."

    def generate_data(self, n):
        A = uniform.rvs(size=n)
        B = uniform.rvs(size=n)
        epsilon = uniform.rvs(size=n)
        assert A.shape == B.shape == epsilon.shape, \
            "Random variables must be the same shape"

        def calculate_c(a, b):
            c = a + b - 1 if a + b > 1 else a + b
            return c
        C = np.vectorize(calculate_c)(A, B)
        C += epsilon
        return (A, B, C)

    def build_vector(self, x, i, d):
        v_x = uniform.rvs(size=d)
        v_x[i] = x
        return v_x

    def get_vectors(self, A, B, i, d):
        v_a = torch.tensor([self.build_vector(a, i, d) for a in A], dtype=torch.float32)
        v_b = torch.tensor([self.build_vector(b, i, d) for b in B], dtype=torch.float32)
        return v_a, v_b

    def get_representations(self, model, v_a, v_b):
        model.eval()
        with torch.no_grad():
            return model(v_a, v_b)

    def __len__(self):
        return len(self.r_a)

    def __getitem__(self, idx):
        r_a = self.r_a[idx, :]
        r_b = self.r_b[idx, :]
        C = self.C[idx]
        return r_a, r_b, C

class PretrainingDataset(Dataset):
    def __init__(self, d, n):
            data = self.generate_data(n)
            i = np.random.randint(0, d)
            self.v_a, self.v_b, self.v_c = self.get_vectors(data, i, d)
            assert self.v_a.shape == self.v_b.shape == self.v_c.shape, \
                "All vectors must be the same shape."

    def generate_data(self, n):
        A = uniform.rvs(size=n)
        B = uniform.rvs(size=n)
        epsilon = uniform.rvs(size=n)
        assert A.shape == B.shape == epsilon.shape, \
            "Random variables must be the same shape"

        def calculate_c(a, b):
            c = a + b - 1 if a + b > 1 else a + b
            return c
        C = np.vectorize(calculate_c)(A, B)
        C += epsilon
        return (A, B, C)

    def build_vector(self, x, i, d):
        v_x = uniform.rvs(size=d)
        v_x[i] = x
        return v_x

    def get_vectors(self, data, i, d):
        A, B, C = data
        v_a = torch.tensor([self.build_vector(a, i, d) for a in A], dtype=torch.float32)
        v_b = torch.tensor([self.build_vector(b, i, d) for b in B], dtype=torch.float32)
        v_c = torch.tensor([self.build_vector(c, i, d) for c in C], dtype=torch.float32)
        return v_a, v_b, v_c

    def __len__(self):
        return len(self.v_a)

    def __getitem__(self, idx):
        v_a = self.v_a[idx, :]
        v_b = self.v_b[idx, :]
        v_c = self.v_c[idx, :]
        return v_a, v_b, v_c