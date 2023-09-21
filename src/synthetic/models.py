from argparse import Namespace

import lightning.pytorch as pl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from src.losses import pairwise_infonce, symile
from src.utils import l2_normalize


class LinearEncoders(nn.Module):
    def __init__(self, d_v, d_r, hardcode_encoders):
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

    def forward(self, v_a, v_b, v_c):
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
        r_c = self.f_c(v_c)
        assert r_a.shape == r_b.shape == r_c.shape, \
            "Representations must be the same shape."
        assert r_a.shape[1] == self.d_r, \
            f"Representations must have dimensionality d_r ({self.d_r})."
        return r_a, r_b, r_c


class XORModule(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)
        self.loss_fn = symile if self.args.loss_fn == "symile" else pairwise_infonce

        self.encoders = LinearEncoders(self.args.d_v, self.args.d_r,
                                       self.args.hardcode_encoders)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

    def forward(self, x):
        v_a, v_b, v_c = x
        r_a, r_b, r_c = self.encoders(v_a, v_b, v_c)
        return r_a, r_b, r_c, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr)

    def _shared_step(self, batch, batch_idx):
        r_a, r_b, r_c, logit_scale_exp = self(batch)

        if self.args.normalize:
            r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])

        return self.loss_fn(r_a, r_b, r_c, logit_scale_exp), logit_scale_exp

    def training_step(self, batch, batch_idx):
        loss, logit_scale_exp = self._shared_step(batch, batch_idx)
        log_n_minus_1 = np.log(len(batch[0])-1)

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n_minus_1", log_n_minus_1,
                 on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, batch_idx)
        log_n_minus_1 = np.log(len(batch[0])-1)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("log_n_minus_1", log_n_minus_1,
                 on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
        return loss

    def get_query_representations(self):
        """
        Get representations for the four query vectors [0,0], [0,1], [1,0], [1,1].
        Returns:
            q (torch.Tensor): query representations of size (4, d_r) where
                              q[0] = f([0,0]), q[1] = f([0,1]),
                              q[2] = f([1,0]), q[3] = f([1,1]).
        """
        q_00 = self.encoders.f_c(torch.Tensor([0,0]).to(self.device))
        q_01 = self.encoders.f_c(torch.Tensor([0,1]).to(self.device))
        q_10 = self.encoders.f_c(torch.Tensor([1,0]).to(self.device))
        q_11 = self.encoders.f_c(torch.Tensor([1,1]).to(self.device))

        assert torch.ne(q_00, q_01).any() and \
               torch.ne(q_00, q_10).any() and \
               torch.ne(q_00, q_11).any() and \
               torch.ne(q_01, q_10).any() and \
               torch.ne(q_01, q_11).any() and \
               torch.ne(q_10, q_11).any(), \
               "q_00, q_01, q_10, q_11 must all be different."

        q = torch.cat((torch.unsqueeze(q_00, 0), torch.unsqueeze(q_01, 0),
                       torch.unsqueeze(q_10, 0), torch.unsqueeze(q_11, 0)), dim=0)

        if self.args.normalize:
            [q] = l2_normalize([q])

        return q

    def on_test_start(self):
        if self.args.evaluation == "zeroshot":
            self.q = self.get_query_representations()

    def zeroshot_step(self, batch):
        r_a, r_b, r_c, logit_scale_exp = self(batch)
        if self.args.normalize:
            r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])

        # get predictions
        if self.args.loss_fn == "symile":
            # logits is a (batch_sz, 4) matrix where each row i is
            # [ MIP(r_a[i], r_b[i], q[0]) ... MIP(r_a[i], r_b[i], q[3]) ]
            # where MIP is the multilinear inner product.
            logits = (r_a * r_b) @ torch.t(self.q)
        elif self.args.loss_fn == "pairwise_infonce":
            # logits is a (batch_sz, 4) matrix where each row i is
            # [ r_a[i]^T r_b[i] + r_b[i]^T q[0] + r_a[i]^T q[0] ...
            #   r_a[i]^T r_b[i] + r_b[i]^T q[3] + r_a[i]^T q[3] ]
            ab = torch.diagonal(r_a @ torch.t(r_b)).unsqueeze(dim=1) # (batch_sz, 1)
            logits = ab + (r_b @ torch.t(self.q)) + (r_a @ torch.t(self.q))

        if self.args.use_logit_scale_eval:
            logits = logit_scale_exp * logits

        preds = torch.argmax(logits, dim=1)

        # get labels
        def _get_label(r):
            return torch.argmax(torch.where(r == self.q, 1, 0).sum(dim=1))
        labels = torch.vmap(_get_label)(r_c)

        return torch.where(preds == labels, 1, 0).sum() / len(labels)

    def support_step(self, batch):
        v_a, v_b, v_c, y = batch
        r_a, r_b, r_c, logit_scale_exp = self((v_a, v_b, v_c))
        if self.args.normalize:
            r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])

        if self.args.loss_fn == "symile":
            X = r_a * r_b * r_c
        elif self.args.loss_fn == "pairwise_infonce":
            if self.args.concat_infonce:
                X = torch.cat((r_a * r_b, r_b * r_c, r_a * r_c), dim=1)
            else:
                X = (r_a * r_b) + (r_b * r_c) + (r_a * r_c)

        if self.args.use_logit_scale_eval:
            X = logit_scale_exp * X

        X_train, X_test, y_train, y_test = train_test_split(X.cpu(), y.cpu(),
                                                            test_size=0.2)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        return clf.score(X_test, y_test)

    def test_step(self, batch, batch_idx):
        if self.args.evaluation == "zeroshot":
            mean_acc = self.zeroshot_step(batch)
        elif self.args.evaluation == "support":
            mean_acc = self.support_step(batch)

        print("Mean accuracy: ", mean_acc)
        self.log("mean_acc", mean_acc,
                 on_step=False, on_epoch=True, sync_dist=True)