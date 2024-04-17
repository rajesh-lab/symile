from argparse import Namespace
from collections import defaultdict
import json

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from datasets import EvaluationDataset
from symile.losses import clip, symile, zeroshot_retrieval_logits
from symile.cxr_prediction.constants import CHEXPERT_LABELS


class ECGEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


class LabsEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
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
        self.labs_encoder = LabsEncoder(self.args.d)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if hasattr(self.args, "freeze_logit_scale") and self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

    def forward(self, x):
        """
        Args:
            x: list of four tensors (cxr, ecg, labs_percentiles, labs_missingness)
        """
        if isinstance(x, list):
            r_c = self.cxr_encoder(x[0])

            r_e = self.ecg_encoder(x[1])

            labs = torch.cat([x[2], x[3]], dim=1)
            r_l = self.labs_encoder(labs)
        else:
        # query or candidate dataset
            r_c = self.cxr_encoder(x["cxr"].to(self.device))

            r_e = self.ecg_encoder(x["ecg"].to(self.device))

            labs = torch.cat([x["labs_percentiles"], x["labs_missingness"]], dim=1)
            r_l = self.labs_encoder(labs.to(self.device))
        return r_c, r_e, r_l, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        r_c, r_e, r_l, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_e, r_l, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch[0]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp, "log_n": log_n},
                    on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        r_c, r_e, r_l, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_e, r_l, logit_scale_exp, self.args.efficient_loss)

        acc = self.zeroshot_retrieval_accuracy(r_e, r_l, batch, "val")

        self.log_dict({"val_loss": loss, "val_accuracy": acc},
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        r_c, r_e, r_l, logit_scale_exp = self(batch)

        acc = self.zeroshot_retrieval_accuracy(r_e, r_l, batch, "test")

        self.log("test_accuracy", acc, sync_dist=True, prog_bar=True)

        return acc

    def on_validation_epoch_start(self):
        self.save_candidate_cxr_representations("val")

        metrics = self.zeroshot_pathology_retrieval_metrics("val")

        self.log_dict(metrics, sync_dist=True, prog_bar=False)

    def on_test_epoch_start(self):
        self.save_candidate_cxr_representations("test")

        metrics = self.zeroshot_pathology_retrieval_metrics("test")

        self.log_dict(metrics, sync_dist=True, prog_bar=False)

        with open(self.args.save_dir / "metrics.json", 'w') as f:
            json.dump(metrics, f, indent=4)

    def save_candidate_cxr_representations(self, split):
        r_c_list = []
        hadm_id_list = []

        # get dataloader
        if split == "val":
            dl = self.trainer.datamodule.val_dataloader()
        elif split == "test":
            if self.trainer.datamodule is None:
                dl = getattr(self, 'test_dataloader', None)
            else:
                dl = self.trainer.datamodule.test_dataloader()

        # get reps
        for x in dl:
            r_c_list.append(self.cxr_encoder(x[0].to(self.device)))
            hadm_id_list.append(x[4])

        # save reps
        if split == "val":
            self.r_c_val = torch.cat(r_c_list)
            self.r_c_hadm_id_val = torch.cat(hadm_id_list).to(self.device)
        elif split == "test":
            self.r_c_test = torch.cat(r_c_list)
            self.r_c_hadm_id_test = torch.cat(hadm_id_list).to(self.device)

    def zeroshot_retrieval_accuracy(self, r_e, r_l, batch, split):
        if split == "val":
            r_c = self.r_c_val
            r_c_hadm_id = self.r_c_hadm_id_val
        elif split == "test":
            r_c = self.r_c_test
            r_c_hadm_id = self.r_c_hadm_id_test

        logits = zeroshot_retrieval_logits(r_e, r_l, r_c, self.logit_scale.exp(),
                                           self.args.loss_fn)

        # pred_idx is a tensor of length batch_sz where each element is the
        # index of the r_c (across the whole eval set) that maximizes the score.
        pred_idx = torch.argmax(logits, dim=1)

        # for each index in pred_idx, we get the hadm_id that corresponds
        # to the r_c at that index; so pred is a tensor of length batch_sz where
        # each element is the predicted hadm_id
        pred = r_c_hadm_id[pred_idx]

        y = batch[4]

        acc = (torch.sum(y == pred) / len(y)).item()

        return acc

    def get_queries(self, split):
        query_ds = EvaluationDataset(self.args, split, "query")

        r_e = []
        r_l = []
        label_name = []

        for batch in DataLoader(query_ds, batch_size=len(query_ds), shuffle=False):
            r_e.append(self.ecg_encoder(batch["ecg"].to(self.device)))

            labs = torch.cat([batch["labs_percentiles"], batch["labs_missingness"]], dim=1)
            r_l.append(self.labs_encoder(labs.to(self.device)))

            label_name += batch["label_name"]

        r_e = torch.cat(r_e, dim=0)
        r_l = torch.cat(r_l, dim=0)

        assert len(r_e) == len(r_l) == len(query_ds), "r_e, r_l, and query_ds should have the same length"

        return {"r_e": r_e, "r_l": r_l, "label_name": label_name}

    def get_candidates(self, split):
        candidate_ds = EvaluationDataset(self.args, split, "candidate")

        r_c = []
        label_name = []
        label_value = []

        for batch in DataLoader(candidate_ds, batch_size=256, shuffle=False):
            r_c.append(self.cxr_encoder(batch["cxr"].to(self.device)))
            label_name += batch["label_name"]
            label_value.append(batch["label_value"])

        r_c = torch.cat(r_c, dim=0)

        assert len(r_c) == len(candidate_ds), "r_c and candidate_ds should have the same length"

        return {"r_c": r_c,
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

    def zeroshot_pathology_retrieval_metrics(self, split):
        queries = self.get_queries(split)
        candidates = self.get_candidates(split)

        precision_at_k_dict = {}
        acc_at_k_dict = {}

        for label in CHEXPERT_LABELS:
            # get query set for `label`
            label_mask_q = torch.tensor([name == label for name in queries["label_name"]])
            r_e = queries["r_e"][label_mask_q]
            r_l = queries["r_l"][label_mask_q]

            # get candidate set for `label`
            label_mask_c = torch.tensor([name == label for name in candidates["label_name"]])
            r_c = candidates["r_c"][label_mask_c]
            r_c_labels = candidates["label_value"][label_mask_c]

            logits = zeroshot_retrieval_logits(r_e, r_l, r_c, self.logit_scale.exp(),
                                               self.args.loss_fn).cpu()

            for k in [1, 5, 10]:
                # get indices of top k logits
                _, topk_indices = torch.topk(logits, k, dim=1)

                # get positive/negative labels at those indices
                topk_labels = r_c_labels[topk_indices]

                # calculate metrics
                precision_at_k = topk_labels.sum(dim=1) / k
                mean_precision_at_k = precision_at_k.mean().item()

                acc_at_k = torch.any(topk_labels, dim=1).float()
                mean_acc_at_k = torch.mean(acc_at_k).item()

                # save metrics
                precision_at_k_dict[f"{label}_precision_at_{k}"] = mean_precision_at_k
                acc_at_k_dict[f"{label}_acc_at_{k}"] = mean_acc_at_k

        precision_at_k_all_labels = self.get_metric_all_labels(precision_at_k_dict, "precision")
        acc_at_k_all_labels = self.get_metric_all_labels(acc_at_k_dict, "acc")

        metrics = precision_at_k_dict | acc_at_k_dict | precision_at_k_all_labels | acc_at_k_all_labels
        metrics = self.add_split_prefix(metrics, split)

        return metrics