import numpy as np
import torch
import torch.nn as nn


class LinearEncoders(nn.Module):
    def __init__(self, d_v, d_r, logit_scale_init, hardcode_encoders):
        """
        Initialize linear encoders that generate representations r_a, r_b, r_c
        from input vectors v_a, v_b, v_c.

        Args:
            d_v (int): dimensionality for each of the vectors v_a, v_b, v_c.
            d_r (int): dimensionality for each of the representations r_a, r_b, r_c.
            logit_scale_init (float): initial value for logit_scale parameter.
        """
        super().__init__()
        self.d_v = d_v
        self.d_r = d_r
        self.f_a = nn.Linear(d_v, d_r, bias=True)
        self.f_b = nn.Linear(d_v, d_r, bias=True)
        self.f_c = nn.Linear(d_v, d_r, bias=True)
        if hardcode_encoders:
            with torch.no_grad():
                weight_ab = torch.zeros(d_r, d_v).fill_diagonal_(2)
                self.f_a.weight = nn.Parameter(weight_ab, requires_grad=False)
                self.f_b.weight = nn.Parameter(weight_ab, requires_grad=False)
                self.f_a.bias = nn.Parameter(torch.full((d_r,), -1.0), requires_grad=False)
                self.f_b.bias = nn.Parameter(torch.full((d_r,), -1.0), requires_grad=False)
                weight_c = torch.zeros(d_r, d_v).fill_diagonal_(-2)
                self.f_c.weight = nn.Parameter(weight_c, requires_grad=False)
                self.f_c.bias = nn.Parameter(torch.full((d_r,), 1.0), requires_grad=False)
        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

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