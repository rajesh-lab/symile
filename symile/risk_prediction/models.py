from argparse import Namespace
import itertools
import json

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from symile.losses import clip, symile
from symile.risk_prediction.constants import RISK_VECTOR_COLS


class ECGEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, d, bias=True)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        """
        Args:
            input_features (torch.Tensor): shape (batch_sz, 80, 3000)
            attention_mask (torch.Tensor): shape (batch_sz, 3000)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.resnet(x)
        x = self.layer_norm(x)
        return x


class CXREncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, d, bias=True)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        """
        Args:
            input_features (torch.Tensor): shape (batch_sz, 80, 3000)
            attention_mask (torch.Tensor): shape (batch_sz, 3000)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.resnet(x)
        x = self.layer_norm(x)
        return x


class RiskEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, d)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(d)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Args:
            input_features (torch.Tensor): shape (batch_sz, 80, 3000)
            attention_mask (torch.Tensor): shape (batch_sz, 3000)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.gelu(x)
        x = self.fc4(x)
        x = self.layer_norm(x)
        return x


class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)
        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.ecg_encoder = ECGEncoder(self.args.d)
        self.cxr_encoder = CXREncoder(self.args.d)
        self.risk_encoder = RiskEncoder(self.args.d)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if hasattr(self.args, "freeze_logit_scale") and self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        self.candidate_risk_vectors = self.get_candidate_risk_vectors()

    def get_candidate_risk_vectors(self):
        value_ranges = {
            "los_quantile": range(4),  # 0, 1, 2, 3
            "hospital_expire_flag": range(2),  # 0, 1
            "adm_within_30_days": range(2)  # 0, 1
        }
        ranges = [value_ranges[col] for col in RISK_VECTOR_COLS]

        all_combinations = list(itertools.product(*ranges))
        return torch.tensor(all_combinations, dtype=torch.float32)

    def forward(self, x):
        r_c = self.cxr_encoder(x["cxr"])
        r_e = self.ecg_encoder(x["ecg"])
        r_r = self.risk_encoder(x["risk_vector"])
        return r_c, r_e, r_r, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        r_c, r_e, r_r, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_e, r_r, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch["hadm_id"]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                    on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n", log_n,
                on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        r_c, r_e, r_r, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_e, r_r, logit_scale_exp, self.args.efficient_loss)

        metrics = self.get_metrics(r_c, r_e, batch, "val")

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict(metrics, sync_dist=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        r_c, r_e, r_r, logit_scale_exp = self(batch)

        metrics = self.get_metrics(r_c, r_e, batch, "test")

        self.log_dict(metrics, sync_dist=True, prog_bar=False)

        with open(self.args.save_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

    def get_logits(self, r_x, r_y, r_z, logit_scale_exp):
        """
        assumes that r_z is the modality to predict
        """
        if self.args.loss_fn == "symile":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ MIP(r_x[i], r_y[i], r_z[0]) ... MIP(r_x[i], r_y[i], r_z[n-1]) ]
            # where MIP is the multilinear inner product.
            logits = (r_x * r_y) @ torch.t(r_z)
        elif self.args.loss_fn == "clip":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ r_x[i]^T r_z[0] + r_z[0]^T r_y[i]   + r_x[i]^T r_y[i] ...
            #   r_x[i]^T r_z[n-1] + r_z[n-1]^T r_y[i] + r_x[i]^T r_y[i] ]
            xy = torch.diagonal(r_x @ torch.t(r_y)).unsqueeze(dim=1) # (batch_sz, 1)
            logits = xy + (r_x @ torch.t(r_z)) + (r_y @ torch.t(r_z))

        return logit_scale_exp * logits

    def add_split_prefix(self, metrics, split):
        """Add a prefix to all metric names."""
        return {f"{split}_{key}": value for key, value in metrics.items()}

    def get_metrics(self, r_c, r_e, batch, split):
        r_r_candidates = self.risk_encoder(self.candidate_risk_vectors.to(self.device))

        logits = self.get_logits(r_c, r_e, r_r_candidates, self.logit_scale.exp())
        logits = logits.cpu()

        acc_at_k = {}
        for k in [1, 5, 10]:
            # get indices of top k logits
            _, topk_indices = torch.topk(logits, k, dim=1) # (batch_sz, k)

            # map indices to risk vectors
            topk_risk_vectors = self.candidate_risk_vectors[topk_indices] # (batch_sz, k, 3)

            # check if true risk vector is in top k predictions
            true_risk_vector = batch["risk_vector"].unsqueeze(1).cpu()
            matches = (topk_risk_vectors == true_risk_vector).all(dim=2) # (batch_sz, k)
            acc = matches.any(dim=1).float() # (batch_sz)

            # save metric
            mean_acc = torch.mean(acc).item()
            acc_at_k[f"acc_at_{k}"] = mean_acc

        acc_at_k = self.add_split_prefix(acc_at_k, split)

        return acc_at_k