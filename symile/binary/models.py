from argparse import Namespace

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn

from utils import get_vector_support
from symile.losses import clip, symile
from symile.utils import l2_normalize


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

    def forward(self, v_a, v_b, v_c):
        """
        Args:
            v_a, v_b, v_c (torch.Tensor): each of size (n, d_v).
        Returns:
            r_a, r_b, r_c (torch.Tensor): each of size (n, d_r).
        """
        r_a = self.f_a(v_a)
        r_b = self.f_b(v_b)
        r_c = self.f_c(v_c)
        assert r_a.shape == r_b.shape == r_c.shape, \
            "Representations must be the same shape."
        assert r_a.shape[1] == self.d_r, \
            f"Representations must have dimensionality d_r ({self.d_r})."
        return r_a, r_b, r_c


class BinaryModule(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)

        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.encoders = LinearEncoders(self.args.d_v, self.args.d_r)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

    def forward(self, v_a, v_b, v_c):
        r_a, r_b, r_c = self.encoders(v_a, v_b, v_c)
        return r_a, r_b, r_c, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def _shared_step(self, batch, batch_idx):
        v_a, v_b, v_c = batch
        r_a, r_b, r_c, logit_scale_exp = self(v_a, v_b, v_c)

        r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])

        loss = self.loss_fn(r_a, r_b, r_c, logit_scale_exp, self.args.efficient_loss)

        return loss, logit_scale_exp

    def training_step(self, batch, batch_idx):
        loss, logit_scale_exp = self._shared_step(batch, batch_idx)

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, batch_idx)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def get_query_representations(self):
        """
        Get representations for the possible "query" vectors v_b. For example:
        - if d_v == 1, the query vectors are: [0], [1]
        - if d_v == 2, the query vectors are: [0,0], [0,1], [1,0], [1,1]

        Returns:
            v_q (torch.Tensor): query vectors in Tensor of size (2^d, d_v).
            r_q (torch.Tensor): query representations of size (2^d, d_r). For
                                example, if d = 2:
                                    r_q[0] = f([0,0]), r_q[1] = f([0,1]),
                                    r_q[2] = f([1,0]), r_q[3] = f([1,1]).
        """
        v_q = get_vector_support(self.args.d_v)
        v_q = [torch.tensor(v) for v in v_q]
        v_q = torch.stack(v_q, dim=0).to(torch.float32).to(self.device)

        r_q = self.encoders.f_b(v_q)

        [r_q] = l2_normalize([r_q])

        return v_q, r_q

    def on_test_start(self):
        """
        The task is to predict which v_b corresponds to a given v_a, v_c.
        At test start, we'll compute the representations for each of the
        possible "query" vectors v_b. For example:
        - if d_v == 1, the query vectors are: [0], [1]
        - if d_v == 2, the query vectors are: [0,0], [0,1], [1,0], [1,1]
        """
        self.v_q, self.r_q = self.get_query_representations()

    def zeroshot_step(self, batch):
        """
        The zeroshot task is to predict which v_b corresponds to a given v_a, v_c.
        """
        v_a, v_b, v_c = batch
        r_a, r_b, r_c, logit_scale_exp = self(v_a, v_b, v_c)

        r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])

        loss = self.loss_fn(r_a, r_b, r_c, logit_scale_exp, self.args.efficient_loss)

        self.log("test_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        # get predictions
        if self.args.loss_fn == "symile":
            # logits is a (batch_sz, 2^d) matrix where each row i is
            # [ MIP(r_a[i], r_q[0], r_c[i]) ... MIP(r_a[i], r_q[2^d - 1], r_c[i]) ]
            # where MIP is the multilinear inner product.
            logits = (r_a * r_c) @ torch.t(self.r_q)
        elif self.args.loss_fn == "clip":
            # logits is a (batch_sz, 2^d) matrix where each row i is
            # [ r_a[i]^T r_c[i] + r_a[i]^T r_q[0]       + r_c[i]^T r_q[0] ...
            #   r_a[i]^T r_c[i] + r_a[i]^T r_q[2^d - 1] + r_c[i]^T r_q[2^d - 1] ]
            ac = torch.diagonal(r_a @ torch.t(r_c)).unsqueeze(dim=1) # (batch_sz, 1)
            logits = ac + (r_a @ torch.t(self.r_q)) + (r_c @ torch.t(self.r_q))

        logits = logit_scale_exp * logits

        preds = torch.argmax(logits, dim=1)

        # get labels
        def _get_label(r):
            return torch.argmax(torch.where(r == self.v_q, 1, 0).sum(dim=1))
        if self.args.d_v == 1:
            labels = torch.squeeze(v_b)
        else:
            labels = torch.vmap(_get_label)(v_b)

        return torch.where(preds == labels, 1, 0).sum() / len(labels)

    def test_step(self, batch, batch_idx):
        acc = self.zeroshot_step(batch)

        self.log("mean_acc", acc, on_step=False, on_epoch=True, sync_dist=True)

        return acc