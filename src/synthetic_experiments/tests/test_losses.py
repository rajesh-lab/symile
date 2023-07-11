import pytest
import torch

from ..losses import *


@pytest.fixture
def r_a():
    return torch.tensor([[-0.3, -0.8],
                         [0.4, -0.7],
                         [-0.4, 0.5]])

@pytest.fixture
def r_b():
    return torch.tensor([[-0.4, -0.4],
                         [-0.6, -0.4],
                         [-0.5, -0.4]])

@pytest.fixture
def r_c():
    return torch.tensor([[0.7, -0.6],
                         [1.2, -0.6],
                         [1.3, 0.7]])

@pytest.fixture
def logit_scale():
    return torch.tensor(14.2)

####################
# pairwise infonce #
####################
def test_infonce(r_a, r_b, logit_scale):
    loss = infonce(r_a, r_b, logit_scale)
    assert loss == pytest.approx(2.9641, abs=1e-4)

def test_pairwise_infonce_normalize_true(r_a, r_b, r_c, logit_scale):
    loss = pairwise_infonce(r_a, r_b, r_c, logit_scale, normalize=True)
    assert loss == pytest.approx(3.3318, abs=1e-4)

def test_pairwise_infonce_normalize_false(r_a, r_b, r_c, logit_scale):
    loss = pairwise_infonce(r_a, r_b, r_c, logit_scale, normalize=False)
    assert loss == pytest.approx(2.7218, abs=1e-4)