from argparse import Namespace
from collections import defaultdict
import json
from json import JSONEncoder
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from datasets import CXRPredictionDataset
from symile.losses import clip, symile, zeroshot_retrieval_logits
from symile.cxr_prediction_age_sex.constants import CHEXPERT_LABELS


class PathToStrEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)     # convert Path object to string
        elif isinstance(obj, Namespace):
            return vars(obj)    # convert Namespace object to dictionary
        return JSONEncoder.default(self, obj)  # default method


class CXREncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.pretrained:
            self.resnet = models.resnet50(weights="IMAGENET1K_V2")
        else:
            self.resnet = models.resnet50(pretrained=False)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, args.d, bias=True)

        self.layer_norm = nn.LayerNorm(args.d)

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


class AgeEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.age_embedding = nn.Embedding(100, args.d)
        self.fc = nn.Linear(args.d, args.d, bias=True)
        self.layer_norm = nn.LayerNorm(args.d)

    def forward(self, x):
        x = self.age_embedding(x)
        x = self.fc(x)
        x = self.layer_norm(x)
        return x


class GenderEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gender_embedding = nn.Embedding(2, args.d)
        self.fc = nn.Linear(args.d, args.d, bias=True)
        self.layer_norm = nn.LayerNorm(args.d)

    def forward(self, x):
        x = self.gender_embedding(x)
        x = self.fc(x)
        x = self.layer_norm(x)
        return x


class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)
        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.cxr_encoder = CXREncoder(self.args)
        self.age_encoder = AgeEmbedding(self.args)
        self.gender_encoder = GenderEmbedding(self.args)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if hasattr(self.args, "freeze_logit_scale") and self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        # logging attributes
        self.run_info = {}

    def forward(self, x):
        r_c = self.cxr_encoder(x["cxr"].to(self.device))
        r_a = self.age_encoder(x["age"].to(self.device))
        r_g = self.gender_encoder(x["gender"].to(self.device))
        return r_c, r_a, r_g, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        r_c, r_a, r_g, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_a, r_g, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch["cxr"]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp, "log_n": log_n},
                    on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        r_c, r_a, r_g, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_a, r_g, logit_scale_exp, self.args.efficient_loss)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        metrics = self.zeroshot_retrieval_accuracy("val")

        self.log_dict(metrics, sync_dist=True, prog_bar=False)

        val_metrics = {
            "epoch": self.current_epoch,
            "val_loss": self.trainer.logged_metrics["val_loss_epoch"].item()
        }

        val_metrics.update({key: value for key, value in metrics.items()})

        self.run_info.setdefault("validation_metrics", []).append(val_metrics)

    def on_train_end(self):
        self.run_info["args"] = self.args

        try:
            self.run_info["wandb"] = self.trainer.logger.experiment.url
        except AttributeError:
            self.run_info["wandb"] = None

        with open(self.args.save_dir / "run_info.json", "w") as f:
            json.dump(self.run_info, f, indent=4, cls=PathToStrEncoder)

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_start(self):
        metrics = self.zeroshot_retrieval_accuracy("test")

        self.log_dict(metrics, sync_dist=True, prog_bar=False)

    def get_queries(self, split):
        query_ds = CXRPredictionDataset(self.args, split, "query")

        r_a = []
        r_g = []
        dicom_id = []
        label_name = []

        for batch in DataLoader(query_ds, batch_size=len(query_ds), shuffle=False):
            r_a.append(self.age_encoder(batch["age"].to(self.device)))
            r_g.append(self.gender_encoder(batch["gender"].to(self.device)))
            dicom_id += batch["dicom_id"]
            label_name += batch["label_name"]

        r_a = torch.cat(r_a, dim=0)
        r_g = torch.cat(r_g, dim=0)

        assert len(r_a) == len(r_g) == len(query_ds), "r_a, r_g, and query_ds should have the same length"

        return {"r_a": r_a, "r_g": r_g, "dicom_id": dicom_id,
                "label_name": label_name}

    def get_candidates(self, split):
        candidate_ds = CXRPredictionDataset(self.args, split, "candidate")

        r_c = []
        dicom_id = []
        label_name = []
        label_value = []

        for batch in DataLoader(candidate_ds, batch_size=256, shuffle=False):
            r_c.append(self.cxr_encoder(batch["cxr"].to(self.device)))
            dicom_id += batch["dicom_id"]
            label_name += batch["label_name"]
            label_value.append(batch["label_value"])

        r_c = torch.cat(r_c, dim=0)

        assert len(r_c) == len(candidate_ds), "r_c and candidate_ds should have the same length"

        return {"r_c": r_c,
                "dicom_id": dicom_id,
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

    def get_clip_logits(self, r_x, r_y, r_z, logit_scale_exp):
        """
        assumes that r_z is the modality to predict
        """
        # logits is a (batch_sz, n) matrix where each row i is
        # [ r_x[i]^T r_z[0] + r_z[0]^T r_y[i]   + r_x[i]^T r_y[i] ...
        #   r_x[i]^T r_z[n-1] + r_z[n-1]^T r_y[i] + r_x[i]^T r_y[i] ]
        xy = torch.diagonal(r_x @ torch.t(r_y)).unsqueeze(dim=1) # (batch_sz, 1)
        xz = r_x @ torch.t(r_z)
        yz = r_y @ torch.t(r_z)

        return (logit_scale_exp * xy,
                logit_scale_exp * xz,
                logit_scale_exp * yz)

    def zeroshot_retrieval_accuracy(self, split):
        queries = self.get_queries(split)
        candidates = self.get_candidates(split)

        precision_at_k_dict = {}
        acc_at_k_dict = {}
        true_cxr_acc_at_k_dict = {}

        for label in CHEXPERT_LABELS:
            # get query set for `label`
            label_mask_q = torch.tensor([name == label for name in queries["label_name"]])
            r_a = queries["r_a"][label_mask_q]
            r_g = queries["r_g"][label_mask_q]

            # get candidate set for `label`
            label_mask_c = torch.tensor([name == label for name in candidates["label_name"]])
            r_c = candidates["r_c"][label_mask_c]
            r_c_labels = candidates["label_value"][label_mask_c]

            # get dicom_ids for query and candidate sets
            all_dicom_ids = queries["dicom_id"] + candidates["dicom_id"]
            dicom_id_to_idx = {dicom_id: idx for idx, dicom_id in enumerate(set(all_dicom_ids))}

            true_dicom_ids = torch.tensor([dicom_id_to_idx[id] for id in queries["dicom_id"]])
            true_dicom_ids = true_dicom_ids[label_mask_q]

            r_c_dicom_id = torch.tensor([dicom_id_to_idx[id] for id in candidates["dicom_id"]])
            r_c_dicom_id = r_c_dicom_id[label_mask_c]

            # logits are (query_sz, candidate_sz) or (5, 125)
            logits = zeroshot_retrieval_logits(r_a, r_g, r_c, self.logit_scale.exp(),
                                               self.args.loss_fn).cpu()

            ## BEGIN TEMP
            # el, ec, lc = self.get_clip_logits(r_a, r_g, r_c, self.logit_scale.exp())
            # el = el.cpu()
            # ec = ec.cpu()
            # lc = lc.cpu()

            # torch.save(logits, self.args.save_dir / "logits.pt")
            # torch.save(el, self.args.save_dir / "el.pt")
            # torch.save(ec, self.args.save_dir / "ec.pt")
            # torch.save(lc, self.args.save_dir / "lc.pt")

            # logits = lc
            ## END TEMP

            for k in [1, 5, 10]:
                # get indices of top k logits
                _, topk_indices = torch.topk(logits, k, dim=1) # (query_sz, k)

                ## PATHOLOGY LABEL METRICS ##

                # get positive/negative labels at those indices
                topk_labels = r_c_labels[topk_indices] # (query_sz, k)

                # calculate metrics
                precision_at_k = topk_labels.sum(dim=1) / k
                mean_precision_at_k = precision_at_k.mean().item()

                acc_at_k = torch.any(topk_labels, dim=1).float()
                mean_acc_at_k = torch.mean(acc_at_k).item()

                # save metrics
                precision_at_k_dict[f"{label}_precision_at_{k}"] = mean_precision_at_k
                acc_at_k_dict[f"{label}_acc_at_{k}"] = mean_acc_at_k

                ## TRUE CXR RETRIEVAL METRIC ##

                # map indices to dicom_ids
                topk_dicom_ids = r_c_dicom_id[topk_indices] # (query_sz, k)

                # check if true dicom_id is in top k predicted dicom_ids
                true_cxr_acc_at_k = (true_dicom_ids.unsqueeze(1) == topk_dicom_ids).any(dim=1).float()
                mean_true_cxr_acc_at_k = torch.mean(true_cxr_acc_at_k).item()

                # save metric
                true_cxr_acc_at_k_dict[f"{label}_true_cxr_acc_at_{k}"] = mean_true_cxr_acc_at_k


        label_precision_at_k_mean = self.get_metric_all_labels(precision_at_k_dict, "label_precision")
        label_acc_at_k_mean = self.get_metric_all_labels(acc_at_k_dict, "label_acc")
        true_cxr_acc_at_k_mean = self.get_metric_all_labels(true_cxr_acc_at_k_dict, "true_cxr_acc")

        metrics = precision_at_k_dict | acc_at_k_dict | true_cxr_acc_at_k_dict | \
                  label_precision_at_k_mean | label_acc_at_k_mean | true_cxr_acc_at_k_mean
        metrics = self.add_split_prefix(metrics, split)

        return metrics