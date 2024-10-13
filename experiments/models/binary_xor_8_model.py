from argparse import Namespace

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn

from src.losses import clip, symile, zeroshot_retrieval_logits
from src.utils import get_vector_support, l2_normalize


class LinearEncoders(nn.Module):
    def __init__(self, d_r):
        """
        Initialize linear encoders that generate representations r_a, r_b, ..., r_j
        from input vectors v_a, v_b, ..., v_j.

        Args:
            d_r (int): dimensionality for each of the representations r_a, r_b, ..., r_j.
        """
        super().__init__()
        self.d_r = d_r
        self.f_a = nn.Linear(1, d_r, bias=True)
        self.f_b = nn.Linear(1, d_r, bias=True)
        self.f_c = nn.Linear(1, d_r, bias=True)
        self.f_d = nn.Linear(1, d_r, bias=True)
        self.f_e = nn.Linear(1, d_r, bias=True)
        self.f_f = nn.Linear(1, d_r, bias=True)
        self.f_g = nn.Linear(1, d_r, bias=True)
        self.f_h = nn.Linear(1, d_r, bias=True)

    def forward(self, v_a, v_b, v_c, v_d, v_e, v_f, v_g, v_h):
        """
        Args:
            v_a, v_b, ..., v_h (torch.Tensor): each of size (n, 1), where n is the batch size.
        Returns:
            r_a, r_b, ..., r_h (torch.Tensor): each of size (n, d_r).
        """
        r_a = self.f_a(v_a)
        r_b = self.f_b(v_b)
        r_c = self.f_c(v_c)
        r_d = self.f_d(v_d)
        r_e = self.f_e(v_e)
        r_f = self.f_f(v_f)
        r_g = self.f_g(v_g)
        r_h = self.f_h(v_h)

        return r_a, r_b, r_c, r_d, r_e, r_f, r_g, r_h


class BinaryXOR8Model(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()

        self.save_hyperparameters()

        self.args = Namespace(**args)

        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.encoders = LinearEncoders(self.args.d)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        if self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        # for logging attributes and metrics
        self.val_step_accuracies = []
        self.test_step_accuracies = []

    def forward(self, inputs):
        """
        Args:
            inputs (tuple or list): Input tensors (v_a, v_b, ..., v_h).
        Returns:
            outputs (tuple): Output tensors (r_a, r_b, ..., r_h).
            logit_scale (torch.Tensor): The logit scale exponentiated.
        """
        r_outputs = self.encoders(*inputs)

        return r_outputs, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def _shared_step(self, batch, batch_idx):
        r_outputs, logit_scale_exp = self(batch)

        r_outputs = l2_normalize(r_outputs)

        loss = self.loss_fn(r_outputs, logit_scale_exp, self.args.negative_sampling)

        return loss, logit_scale_exp

    def training_step(self, batch, batch_idx):
        loss, logit_scale_exp = self._shared_step(batch, batch_idx)

        log_n = np.log(len(batch[0]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp, "log_n": log_n},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        """
        The task is to predict which v_a corresponds to a given v_b, ..., v_h.
        At test start, we'll compute the representations for each of the
        possible candidate vectors v_a, which are [0], [1].
        """
        assert self.val_step_accuracies == [], "val_step_accuracies is not empty"

        self.v_a_val, self.r_a_val = self.get_candidate_representations()

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_step(batch, batch_idx)

        accuracies = self.zeroshot_retrieval(batch, "val")

        self.val_step_accuracies.extend(accuracies)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Calculates mean test accuracy from the recorded step accuracies and logs
        mean accuracy. It also clears the list of val step accuracies for the
        next epoch.
        """
        mean_acc = sum(self.val_step_accuracies) / len(self.val_step_accuracies)

        self.log("val_acc", mean_acc, sync_dist=True, prog_bar=True)

        self.val_step_accuracies.clear()

    def on_test_start(self):
        """
        The task is to predict which v_a corresponds to a given v_b, ..., v_h.
        At test start, we'll compute the representations for each of the
        possible candidate vectors v_a, which are [0], [1].
        """
        assert self.test_step_accuracies == [], "test_step_accuracies is not empty"

        self.v_a_test, self.r_a_test = self.get_candidate_representations()

    def test_step(self, batch, batch_idx):
        accuracies = self.zeroshot_retrieval(batch, "test")

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
        Get representations for the possible candidate vectors v_a, which are
        [0], [1].

        Returns:
            v_a (torch.Tensor): candidate vectors in a tensor [[0.],[1.]] with
                                shape (2, 1).
            r_a (torch.Tensor): candidate representations with shape (2, d_r).
        """
        v_a = get_vector_support(1)
        v_a = [torch.tensor(v) for v in v_a]
        v_a = torch.stack(v_a, dim=0).to(torch.float32).to(self.device)

        r_a = self.encoders.f_a(v_a)

        [r_a] = l2_normalize([r_a])

        return v_a, r_a

    def zeroshot_retrieval(self, batch, split):
        """
        The zeroshot task is to predict which v_a corresponds to a given v_b, ..., v_h.
        """
        r_a_candidates = self.r_a_val if split == "val" else self.r_a_test

        r_outputs, _ = self(batch)
        r_a, r_b, r_c, r_d, r_e, r_f, r_g, r_h = l2_normalize(r_outputs)

        # logits is a tensor of shape (batch_sz, 2) where each element in a
        # row is the score for the corresponding v_b, ..., v_j
        logits = zeroshot_retrieval_logits(
            r_a_candidates,
            [r_b, r_c, r_d, r_e, r_f, r_g, r_h],
            self.logit_scale.exp(),
            self.args.loss_fn
        )

        preds = torch.argmax(logits, dim=1)

        labels = torch.squeeze(batch[0])

        accuracies = torch.where(preds == labels, 1, 0).float().tolist()

        return accuracies