from argparse import Namespace
from collections import defaultdict
import json

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from symile.losses import clip, symile


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

    def forward(self, x):
        r_c = self.cxr_encoder(x["cxr"])
        r_e = self.ecg_encoder(x["ecg"])
        r_r = self.risk_encoder(x["risk_vector"])
        return r_c, r_e, r_r, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _shared_step(self, batch):
        r_c, r_e, r_r, logit_scale_exp = self(batch)
        return r_c, r_e, r_r, logit_scale_exp

    def training_step(self, batch, batch_idx):
        r_c, r_e, r_r, logit_scale_exp = self._shared_step(batch)

        loss = self.loss_fn(r_c, r_e, r_r, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch["hadm_id"]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                    on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n", log_n,
                on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    # def on_validation_epoch_start(self):
    #     query_ds = EvaluationDataset(self.args, "val", "query")

    #     i = 0
    #     for batch in DataLoader(query_ds, batch_size=len(query_ds), shuffle=False):
    #         r_e, r_c, r_l, logit_scale_exp = self._shared_step(batch)
    #         i += 1
    #     assert i == 1, "query_ds should only have one batch"

    #     metrics = self.get_metrics(r_e, r_l, batch, "val")
    #     self.log_dict(metrics["metrics"], sync_dist=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        r_e, r_c, r_l, logit_scale_exp = self._shared_step(batch)

        loss = self.loss_fn(r_e, r_c, r_l, logit_scale_exp, self.args.efficient_loss)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        r_e, r_c, r_l, logit_scale_exp = self._shared_step(batch)

        metrics = self.get_metrics(r_e, r_l, batch, "test")

        self.log_dict(metrics["metrics"], sync_dist=True, prog_bar=False)
        self.log_dict(metrics["metrics_per_labels"], sync_dist=True, prog_bar=False)

        with open(self.args.save_dir / "metrics.json", 'w') as f:
            json.dump(metrics["metrics"], f, indent=4)
        with open(self.args.save_dir / "metrics_per_labels.json", 'w') as f:
            json.dump(metrics["metrics_per_labels"], f, indent=4)

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

    def get_logits_two_modes(self, r_x, r_y, logit_scale_exp):
        """
        assumes that r_z is the modality to predict
        """
        if self.args.loss_fn == "symile":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ MIP(r_x[i], r_y[i], r_z[0]) ... MIP(r_x[i], r_y[i], r_z[n-1]) ]
            # where MIP is the multilinear inner product.
            logits = r_x @ r_y.T
        elif self.args.loss_fn == "clip":
            ValueError("This function is only for symile loss")

        return logit_scale_exp * logits

    def get_candidates(self, split):
        candidate_ds = EvaluationDataset(self.args, split, "candidate")

        r_c = []
        hadm_id = []
        label_name = []
        label_value = []

        for batch in DataLoader(candidate_ds, batch_size=256, shuffle=False):
            r_c.append(self.cxr_encoder(batch["cxr"].to(self.device)))
            hadm_id.append(batch["hadm_id"])
            label_name += batch["label_name"]
            label_value.append(batch["label_value"])

        r_c = torch.cat(r_c, dim=0)

        assert len(r_c) == len(candidate_ds), "r_c and candidate_ds should have the same length"

        return {"r_c": r_c,
                "hadm_id": torch.cat(hadm_id, dim=0),
                "label_name": label_name,
                "label_value": torch.cat(label_value, dim=0)}

    def get_metric_all_labels(self, metric_dict, metric_name):
        # calculcate metrics across all labels
        metrics = defaultdict(list)
        for key, value in metric_dict.items():
            _, k = key.rsplit("_at_", 1)
            metrics[k].append(value)

        mean_metrics = {}
        for k, values in metrics.items():
            mean_metrics[f"{metric_name}_at_{k}"] = sum(values) / len(values)

        return mean_metrics

    def add_split_prefix(self, metrics, split):
        """Add a prefix to all metric names."""
        return {f"{split}_{key}": value for key, value in metrics.items()}

    def get_metrics(self, r_e_query, r_l_query, batch, split):
        candidates = self.get_candidates(split)

        precision_at_k_dict = {}
        acc_at_k_dict = {}
        acc_at_k_true_cxr_dict = {}

        for label in CHEXPERT_LABELS:
            # get query set for `label`
            label_mask_q = torch.tensor([name == label for name in batch["label_name"]])
            r_e = r_e_query[label_mask_q]
            r_l = r_l_query[label_mask_q]
            true_hadm_id = batch['hadm_id'][label_mask_q].cpu()

            # get candidate set for `label`
            label_mask_c = torch.tensor([name == label for name in candidates["label_name"]])
            r_c = candidates["r_c"][label_mask_c]
            r_c_labels = candidates["label_value"][label_mask_c]
            r_c_hadm_id = candidates["hadm_id"][label_mask_c]

            logits = self.get_logits(r_e, r_l, r_c, self.logit_scale.exp()).cpu()
            # logits = self.get_logits_two_modes(r_e, r_c, self.logit_scale.exp()).cpu()

            for k in [1, 5, 10, 50, 100]:
                # get indices of top k logits
                _, topk_indices = torch.topk(logits, k, dim=1)

                # get positive/negative labels at those indices
                topk_labels = r_c_labels[topk_indices]

                # get hadm_id of top k
                topk_hadm_id = r_c_hadm_id[topk_indices]

                # calculate metrics
                precision_at_k = topk_labels.sum(dim=1) / k
                mean_precision_at_k = precision_at_k.mean().item()

                acc_at_k = torch.any(topk_labels, dim=1).float()
                mean_acc_at_k = torch.mean(acc_at_k).item()

                acc_at_k_true_cxr = (true_hadm_id.unsqueeze(1) == topk_hadm_id).any(dim=1).float()
                mean_acc_at_k_true_cxr = acc_at_k_true_cxr.mean().item()

                # save metrics
                precision_at_k_dict[f"{label}_precision_at_{k}"] = mean_precision_at_k
                acc_at_k_dict[f"{label}_acc_at_{k}"] = mean_acc_at_k
                acc_at_k_true_cxr_dict[f"{label}_acc_at_{k}_true_cxr"] = mean_acc_at_k_true_cxr

        precision_at_k_all_labels = self.get_metric_all_labels(precision_at_k_dict, "precision")
        acc_at_k_all_labels = self.get_metric_all_labels(acc_at_k_dict, "acc")
        acc_at_k_true_cxr_all_labels = self.get_metric_all_labels(acc_at_k_true_cxr_dict, "acc_true_cxr")

        metrics_per_labels = precision_at_k_dict | acc_at_k_dict | acc_at_k_true_cxr_dict
        metrics = precision_at_k_all_labels | acc_at_k_all_labels | acc_at_k_true_cxr_all_labels

        metrics_per_labels = self.add_split_prefix(metrics_per_labels, split)
        metrics = self.add_split_prefix(metrics, split)

        return {"metrics_per_labels": metrics_per_labels, "metrics": metrics}