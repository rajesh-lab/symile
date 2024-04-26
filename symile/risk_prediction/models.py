from argparse import Namespace
from collections import defaultdict
import itertools
import json
from json import JSONEncoder
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from symile.losses import clip, symile, zeroshot_retrieval_logits
from symile.risk_prediction.constants import RISK_VECTOR_COLS


class PathToStrEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)     # convert Path object to string
        elif isinstance(obj, Namespace):
            return vars(obj)    # convert Namespace object to dictionary
        return JSONEncoder.default(self, obj)  # default method


class ECGEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.pretrained:
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
        else:
            self.resnet = models.resnet18(pretrained=False)

        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


class RiskEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, args.d)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.d)

        # self.dropout = nn.Dropout(0.3)

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
        # x = self.dropout(x)
        x = self.fc2(x)
        x = self.gelu(x)
        # x = self.dropout(x)
        x = self.fc3(x)
        x = self.layer_norm(x)
        return x


class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)
        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.ecg_encoder = ECGEncoder(self.args)
        self.cxr_encoder = CXREncoder(self.args)
        self.risk_encoder = RiskEncoder(self.args)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if hasattr(self.args, "freeze_logit_scale") and self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        self.candidate_risk_vectors = self.get_candidate_risk_vectors()

        # logging attributes
        self.run_info = {}
        self.val_metrics = {}
        self.test_metrics = {}

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
        r_c = self.cxr_encoder(x[0])
        r_e = self.ecg_encoder(x[1])
        r_r = self.risk_encoder(x[2])
        return r_c, r_e, r_r, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        r_c, r_e, r_r, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_e, r_r, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch[3]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                    on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n", log_n,
                on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        r_c, r_e, r_r, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_e, r_r, logit_scale_exp, self.args.efficient_loss)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        metrics = self.zeroshot_retrieval_accuracy(r_c, r_e, batch, "val")

        if not self.val_metrics:
            self.val_metrics = {key: val.detach() for key, val in metrics.items()}
        else:
            for key, val in metrics.items():
                self.val_metrics[key] = torch.cat((self.val_metrics[key], val.detach()), dim=0)

        return loss

    def on_validation_epoch_end(self):
        mean_metrics = {}
        for key, val in self.val_metrics.items():
            mean_metrics[key] = val.mean().item()

        self.log_dict(mean_metrics, on_epoch=True, prog_bar=False)

        self.run_info.setdefault("validation_metrics", []).append({
            "epoch": self.current_epoch,
            "val_loss": self.trainer.logged_metrics["val_loss_epoch"].item(),
            "val_acc_at_1": mean_metrics["val_acc_at_1"],
            "val_acc_at_5": mean_metrics["val_acc_at_5"],
            "val_acc_at_10": mean_metrics["val_acc_at_10"]
        })

        # clear to free up memory and prepare for next val epoch
        self.val_metrics.clear()

    def on_train_end(self):
        self.run_info["args"] = self.args

        try:
            self.run_info["wandb"] = self.trainer.logger.experiment.url
        except AttributeError:
            self.run_info["wandb"] = None

        with open(self.args.save_dir / "run_info.json", "w") as f:
            json.dump(self.run_info, f, indent=4, cls=PathToStrEncoder)

    def test_step(self, batch, batch_idx):
        r_c, r_e, r_r, logit_scale_exp = self(batch)

        metrics = self.zeroshot_retrieval_accuracy(r_c, r_e, batch, "test")

        if not self.test_metrics:
            self.test_metrics = {key: val.detach() for key, val in metrics.items()}
        else:
            for key, val in metrics.items():
                self.test_metrics[key] = torch.cat((self.test_metrics[key], val.detach()), dim=0)

    def on_test_epoch_end(self):
        mean_metrics = {}
        for key, val in self.test_metrics.items():
            mean_metrics[key] = val.mean().item()

        self.log_dict(mean_metrics, on_epoch=True, prog_bar=False)

        # clear to free up memory and prepare for next val epoch
        self.test_metrics.clear()

    def add_split_prefix(self, metrics, split):
        """Add a prefix to all metric names."""
        return {f"{split}_{key}": value for key, value in metrics.items()}

    def zeroshot_retrieval_accuracy(self, r_c, r_e, batch, split):
        r_r_candidates = self.risk_encoder(self.candidate_risk_vectors.to(self.device))

        logits = zeroshot_retrieval_logits(r_c, r_e, r_r_candidates, self.logit_scale.exp(),
                                           self.args.loss_fn)
        logits = logits.cpu()

        acc_at_k = {}
        for k in [1, 5, 10]:
            # get indices of top k logits
            _, topk_indices = torch.topk(logits, k, dim=1) # (batch_sz, k)

            # map indices to risk vectors
            topk_risk_vectors = self.candidate_risk_vectors[topk_indices] # (batch_sz, k, 3)

            # check if true risk vector is in top k predictions
            true_risk_vector = batch[2].unsqueeze(1).cpu()
            matches = (topk_risk_vectors == true_risk_vector).all(dim=2) # (batch_sz, k)
            acc = matches.any(dim=1).float() # (batch_sz)

            # save metric
            # mean_acc = torch.mean(acc).item()
            # acc_at_k[f"acc_at_{k}"] = mean_acc
            acc_at_k[f"acc_at_{k}"] = acc

        acc_at_k = self.add_split_prefix(acc_at_k, split)

        return acc_at_k