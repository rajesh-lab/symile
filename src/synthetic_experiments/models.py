import numpy as np
import torch
import torch.nn as nn


class LinearEncoders(nn.Module):
    def __init__(self, d_v, d_r):
        """
        Initialize linear encoders that generate representations r_a, r_b, r_c
        from input vectors v_a, v_b, v_c.

        Args:
            d_v (int): dimensionality for each of the vectors v_a, v_b, v_c.
            d_r (int): dimensionality for each of the representations r_a, r_b, r_c.
        """
        super().__init__()
        self.d_v = d_v
        self.d_r = d_r
        self.f_a = nn.Linear(d_v, d_r, bias=True)
        self.f_b = nn.Linear(d_v, d_r, bias=True)
        self.f_c = nn.Linear(d_v, d_r, bias=True)
        # temperature parameter used by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, v_a, v_b, v_c=None):
        """
        Args:
            v_a, v_b, v_c (torch.Tensor): each of size (n, d_v) (v_c is optional).
        Returns:
            r_a, r_b, r_c (torch.Tensor): each of size (n, d_r) (v_c could be None).
            self.logit_scale.exp() (torch.Tensor): temperature parameter as a
                a log-parameterized multiplicative scalar (see CLIP).
        """
        r_a = self.f_a(v_a)
        r_b = self.f_b(v_b)
        r_c = self.f_c(v_c) if v_c is not None else None
        assert r_a.shape == r_b.shape, "Representations must be the same shape."
        assert r_a.shape[1] == self.d_r, \
            f"Representations must have dimensionality d_r ({self.d_r})."
        return r_a, r_b, r_c, self.logit_scale.exp()