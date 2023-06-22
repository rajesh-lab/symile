import numpy as np
import torch
import torch.nn as nn


class LinearEncoders(nn.Module):
    def __init__(self, d_v, d_r):
        super().__init__()
        self.f_a = nn.Linear(d_v, d_r, bias=True)
        self.f_b = nn.Linear(d_v, d_r, bias=True)
        self.f_c = nn.Linear(d_v, d_r, bias=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, v_a, v_b, v_c):
        r_a = self.f_a(v_a)
        r_b = self.f_b(v_b)
        r_c = self.f_c(v_c)
        return r_a, r_b, r_c, self.logit_scale.exp()