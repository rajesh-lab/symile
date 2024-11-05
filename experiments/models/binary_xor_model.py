from argparse import Namespace

import lightning.pytorch as pl
import torch
import torch.nn as nn

from losses import clip, symile, zeroshot_retrieval_logits
from utils import get_vector_support, l2_normalize


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


class BinaryXORModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()

        self.save_hyperparameters()

        self.args = Namespace(**args)

        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.encoders = LinearEncoders(self.args.d_v, self.args.d)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        if self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        # for logging test metrics
        self.test_step_accuracies = []

    def forward(self, v_a, v_b, v_c):
        r_a, r_b, r_c = self.encoders(v_a, v_b, v_c)
        return r_a, r_b, r_c, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def _shared_step(self, batch, batch_idx):
        v_a, v_b, v_c = batch
        r_a, r_b, r_c, logit_scale_exp = self(v_a, v_b, v_c)

        r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])

        loss = self.loss_fn(r_a, r_b, r_c, logit_scale_exp, self.args.negative_sampling)

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

    def on_test_start(self):
        """
        The task is to predict which v_b corresponds to a given v_a, v_c.
        At test start, we'll compute the representations for each of the
        possible candidate vectors v_b. For example:
        - if d_v == 1, the candidate vectors are: [0], [1]
        - if d_v == 2, the candidate vectors are: [0,0], [0,1], [1,0], [1,1]
        """
        assert self.test_step_accuracies == [], "test_step_accuracies is not empty"

        self.v_b, self.r_b = self.get_candidate_representations()

    def test_step(self, batch, batch_idx):
        accuracies = self.zeroshot_retrieval(batch)

        self.test_step_accuracies.extend(accuracies)

    def on_test_epoch_end(self):
        """
        Calculates mean test accuracy from the recorded step accuracies and logs
        mean accuracy. It also clears the list of test step accuracies for the
        next epoch.
        """
        mean_acc = sum(self.test_step_accuracies) / len(self.test_step_accuracies)

        self.log("test_acc", mean_acc, sync_dist=True, prog_bar=True)

        self.test_step_accuracies.clear()

    def get_candidate_representations(self):
        """
        Get representations for the possible candidate vectors v_b. For example:
        - if d_v == 1, the candidate vectors are: [0], [1]
        - if d_v == 2, the candidate vectors are: [0,0], [0,1], [1,0], [1,1]

        Returns:
            v_b (torch.Tensor): candidate vectors in Tensor of size (2^d, d_v).
            r_b (torch.Tensor): candidate representations of size (2^d, d_r). For
                                example, if d = 2:
                                    r_q[0] = f([0,0]), r_q[1] = f([0,1]),
                                    r_q[2] = f([1,0]), r_q[3] = f([1,1]).
        """
        v_b = get_vector_support(self.args.d_v)
        v_b = [torch.tensor(v) for v in v_b]
        v_b = torch.stack(v_b, dim=0).to(torch.float32).to(self.device)

        r_b = self.encoders.f_b(v_b)

        [r_b] = l2_normalize([r_b])

        return v_b, r_b

    def zeroshot_retrieval(self, batch):
        """
        The zeroshot task is to predict which v_b corresponds to a given v_a, v_c.
        """
        v_a, v_b, v_c = batch
        r_a, r_b, r_c, _ = self(v_a, v_b, v_c)
        r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])

        # logits is a tensor of shape (batch_sz, 2^d) where each element in a
        # row is the score for the corresponding a and c.
        logits = zeroshot_retrieval_logits(self.r_b, [r_a, r_c], self.logit_scale.exp(),
                                           self.args.loss_fn)

        preds = torch.argmax(logits, dim=1)

        # get labels
        def _get_label(r):
            return torch.argmax(torch.where(r == self.v_b, 1, 0).sum(dim=1))
        if self.args.d_v == 1:
            labels = torch.squeeze(v_b)
        else:
            labels = torch.vmap(_get_label)(v_b)

        accuracies = torch.where(preds == labels, 1, 0).float().tolist()

        return accuracies