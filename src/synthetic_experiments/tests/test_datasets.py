import pytest

from ..datasets import FinetuningDataset, PretrainingDataset

@pytest.fixture
def pretraining_dataset():
    """
    Returns a PretrainingDataset object initialized with dimensionality 2 and
    number of samples 5.
    """
    return PretrainingDataset(2, 5)

def test_adriel_fn(pretraining_dataset):
    (A, B, C) = pretraining_dataset.generate_data(2)
    print(A)
    return A



# write the comments for pretraining dataset