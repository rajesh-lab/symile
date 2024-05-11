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

from symile.losses import clip, symile, zeroshot_retrieval_logits


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


class AgeEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.age_embedding = nn.Embedding(4, args.d)
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

        self.ecg_encoder = ECGEncoder(self.args)
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
        r_e = self.ecg_encoder(x[0].to(self.device))
        r_a = self.age_encoder(x[1].to(self.device))
        r_g = self.gender_encoder(x[2].to(self.device))
        return r_e, r_a, r_g, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        r_e, r_a, r_g, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_e, r_a, r_g, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch[3]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp, "log_n": log_n},
                    on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        self.save_candidate_ecgs("val")

    def on_test_epoch_start(self):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        self.save_candidate_ecgs("test")

    def validation_step(self, batch, batch_idx):
        r_e, r_a, r_g, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_e, r_a, r_g, logit_scale_exp, self.args.efficient_loss)

        metrics = self.zeroshot_retrieval_accuracy(r_a, r_g, batch, "val")

        self.log("val_loss", loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict(metrics, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        """
        The zeroshot task is to predict which r_i corresponds to a given r_a, r_t.
        """
        r_e, r_a, r_g, logit_scale_exp = self(batch)

        acc = self.zeroshot_retrieval_accuracy(r_a, r_g, batch, "test")

        self.log("test_accuracy", acc, sync_dist=True, prog_bar=True)

        return acc

    def on_validation_epoch_end(self):
        val_metrics = {
            "epoch": self.current_epoch,
            "val_loss": self.trainer.logged_metrics["val_loss_epoch"].item(),
            "val_acc_at_1": self.trainer.logged_metrics["val_acc_at_1_epoch"].item(),
            "val_acc_at_5": self.trainer.logged_metrics["val_acc_at_5_epoch"].item(),
            "val_acc_at_10": self.trainer.logged_metrics["val_acc_at_10_epoch"].item(),
            "val_precision_at_1": self.trainer.logged_metrics["val_precision_at_1_epoch"].item(),
            "val_precision_at_5": self.trainer.logged_metrics["val_precision_at_5_epoch"].item(),
            "val_precision_at_10": self.trainer.logged_metrics["val_precision_at_10_epoch"].item()
        }

        self.run_info.setdefault("validation_metrics", []).append(val_metrics)

    def on_train_end(self):
        self.run_info["args"] = self.args

        try:
            self.run_info["wandb"] = self.trainer.logger.experiment.url
        except AttributeError:
            self.run_info["wandb"] = None

        with open(self.args.save_dir / "run_info.json", "w") as f:
            json.dump(self.run_info, f, indent=4, cls=PathToStrEncoder)

    def save_candidate_ecgs(self, split):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        r_e_list = []
        age_list = []
        gender_list = []
        study_id_list = []

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
            r_e_list.append(self.ecg_encoder(x[0].to(self.device)))
            age_list.append(x[1])
            gender_list.append(x[2])
            study_id_list.append(x[3])

        # save reps
        if split == "val":
            self.r_e_val = torch.cat(r_e_list)
            self.age_val = torch.cat(age_list).to(self.device)
            self.gender_val = torch.cat(gender_list).to(self.device)
            self.study_id_val = torch.cat(study_id_list).to(self.device)
        elif split == "test":
            self.r_e_test = torch.cat(r_e_list)
            self.age_test = torch.cat(age_list).to(self.device)
            self.gender_test = torch.cat(gender_list).to(self.device)
            self.study_id_test = torch.cat(study_id_list).to(self.device)

    def add_split_prefix(self, metrics, split):
        """Add a prefix to all metric names."""
        return {f"{split}_{key}": value for key, value in metrics.items()}

    def zeroshot_retrieval_accuracy(self, r_a, r_g, batch, split):
        if split == "val":
            r_e = self.r_e_val
            r_e_age = self.age_val
            r_e_gender = self.gender_val
            r_e_study_id = self.study_id_val
        elif split == "test":
            r_e = self.r_e_test
            r_e_age = self.age_test
            r_e_gender = self.gender_test
            r_e_study_id = self.study_id_test

        logits = zeroshot_retrieval_logits(r_a, r_g, r_e, self.logit_scale.exp(),
                                           self.args.loss_fn)

        true_ages = batch[1]  # (256) - True ages corresponding to each row in logits
        true_genders = batch[2]  # (256) - True genders corresponding to each row in logits

        metrics = {}

        for k in [1, 5, 10]:
            # get indices of top k logits
            topk_indices = torch.topk(logits, k, dim=1).indices # (batch_sz, k)

            # Retrieve the corresponding true ages and genders for each of the top-k indices
            topk_ages = r_e_age[topk_indices]
            topk_genders = r_e_gender[topk_indices]

            correct_matches = (
                (topk_ages == true_ages.unsqueeze(1)) & (topk_genders == true_genders.unsqueeze(1))
            )

            # accuracy at k: check if correct label appears at least once in topk
            acc_at_k = correct_matches.any(dim=1).float()
            mean_acc_at_k = acc_at_k.mean().item()
            metrics[f'acc_at_{k}'] = mean_acc_at_k

            precision_at_k = correct_matches.float().sum(dim=1) / k
            mean_precision_at_k = precision_at_k.mean().item()
            metrics[f'precision_at_{k}'] = mean_precision_at_k

        metrics = self.add_split_prefix(metrics, split)

        return metrics