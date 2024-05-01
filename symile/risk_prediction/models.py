from argparse import Namespace
from collections import defaultdict
import itertools
import json
from json import JSONEncoder
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from rtdl_revisiting_models import FTTransformer
import torch
import torch.nn as nn
from torchvision import models

from symile.losses import clip, symile, zeroshot_retrieval_logits
from symile.risk_prediction.constants import RISK_VECTOR_COLS


def get_risk_vector_ranges():
    value_ranges = {
        "los_quantile": range(4),  # 0, 1, 2, 3
        "hospital_expire_flag": range(2),  # 0, 1
        "adm_within_30_days": range(2)  # 0, 1
    }
    return [value_ranges[col] for col in RISK_VECTOR_COLS]


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


class LabsEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(102, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, args.d)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.d)

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
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x = self.layer_norm(x)
        return x


class RiskEncoderFTT(nn.Module):
    def __init__(self, args):
        super().__init__()
        ranges = get_risk_vector_ranges()
        cat_cardinalities = [len(r) for r in ranges]

        self.model = FTTransformer(
            n_cont_features=0,
            cat_cardinalities=cat_cardinalities,
            d_out=args.d,
            n_blocks=3,
            d_block=192,
            attention_n_heads=8,
            attention_dropout=0.2,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=0.1,
            residual_dropout=0.0,
        )

        self.layer_norm = nn.LayerNorm(args.d)

    def forward(self, x):
        """
        Args:
            input_features (torch.Tensor): shape (batch_sz, 80, 3000)
            attention_mask (torch.Tensor): shape (batch_sz, 3000)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.model(None, x.long())
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
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x = self.layer_norm(x)
        return x


class RiskEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.risk_embedding = nn.Embedding(16, args.d)
        self.fc = nn.Linear(args.d, args.d, bias=True)
        self.layer_norm = nn.LayerNorm(args.d)

    def forward(self, x):
        x = self.risk_embedding(x)
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
        self.labs_encoder = LabsEncoder(self.args)

        if self.args.risk_model == "mlp":
            self.risk_encoder = RiskEncoder(self.args)
        elif self.args.risk_model == "embedding":
            self.risk_encoder = RiskEmbedding(self.args)
        elif self.args.risk_model == "ftt":
            self.risk_encoder = RiskEncoderFTT(self.args)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if hasattr(self.args, "freeze_logit_scale") and self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        self.candidate_risk_vectors = self.get_candidate_risk_vectors()
        self.candidate_risk_vector_to_index = {tuple(vector.numpy()): idx for idx, vector in enumerate(self.candidate_risk_vectors)}

        # logging attributes
        self.run_info = {}
        self.val_metrics = {}
        self.test_metrics = {}

    def get_candidate_risk_vectors(self):
        ranges = get_risk_vector_ranges()
        all_combinations = list(itertools.product(*ranges))
        return torch.tensor(all_combinations, dtype=torch.float32)

    def forward(self, x):
        r_e = self.ecg_encoder(x[0])

        labs = torch.cat([x[1], x[2]], dim=1)
        r_l = self.labs_encoder(labs)

        if self.args.risk_model == "embedding":
            risk_ids = [self.candidate_risk_vector_to_index[tuple(vector.numpy())] for vector in x[3].cpu()]
            risk_ids = torch.tensor(risk_ids, dtype=torch.long).to(self.device)
            r_r = self.risk_encoder(risk_ids)
        else:
            r_r = self.risk_encoder(x[3])

        return r_e, r_l, r_r, self.logit_scale.exp()

    def configure_optimizers(self):
        if self.args.risk_model == "ftt":
            # to protect some parameters from weight decay (per paper). see:
            # https://github.com/yandex-research/rtdl-revisiting-models/tree/2542a25b2adbfd0e9d18ce00f75f9f64ad2c26bd/package#ft-transformer-
            # https://github.com/yandex-research/rtdl-revisiting-models/blob/2542a25b2adbfd0e9d18ce00f75f9f64ad2c26bd/package/rtdl_revisiting_models.py#L780
            param_group_less_risk_encoder = [{"params": [p for name, p in self.named_parameters() if "risk_encoder.model" not in name]}]
            param_groups = param_group_less_risk_encoder + self.risk_encoder.model.make_parameter_groups()
            return torch.optim.AdamW(param_groups, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        r_e, r_l, r_r, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_e, r_l, r_r, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch[4]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                    on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n", log_n,
                on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        r_e, r_l, r_r, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_e, r_l, r_r, logit_scale_exp, self.args.efficient_loss)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        metrics = self.zeroshot_retrieval_accuracy(r_e, r_l, batch, "val")

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
        r_e, r_l, r_r, logit_scale_exp = self(batch)

        metrics = self.zeroshot_retrieval_accuracy(r_e, r_l, batch, "test")

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

    def zeroshot_retrieval_accuracy(self, r_e, r_l, batch, split):
        if self.args.risk_model == "embedding":
            risk_ids = [self.candidate_risk_vector_to_index[tuple(vector.numpy())] for vector in self.candidate_risk_vectors.cpu()]
            risk_ids = torch.tensor(risk_ids, dtype=torch.long).to(self.device)
            r_r_candidates = self.risk_encoder(risk_ids)
        else:
            r_r_candidates = self.risk_encoder(self.candidate_risk_vectors.to(self.device))

        logits = zeroshot_retrieval_logits(r_e, r_l, r_r_candidates, self.logit_scale.exp(),
                                           self.args.loss_fn)
        logits = logits.cpu()

        ## BEGIN TEMP
        # ce, cr, er = self.get_clip_logits(r_c, r_e, r_r_candidates, self.logit_scale.exp())
        # ce = ce.cpu()
        # cr = cr.cpu()
        # er = er.cpu()

        # torch.save(logits, self.args.save_dir / "logits.pt")
        # torch.save(ce, self.args.save_dir / "ce.pt")
        # torch.save(cr, self.args.save_dir / "cr.pt")
        # torch.save(er, self.args.save_dir / "er.pt")

        # logits = er
        ## END TEMP

        acc_at_k = {}
        for k in [1, 5, 10]:
            # get indices of top k logits
            _, topk_indices = torch.topk(logits, k, dim=1) # (batch_sz, k)

            # map indices to risk vectors
            topk_risk_vectors = self.candidate_risk_vectors[topk_indices] # (batch_sz, k, 3)

            # check if true risk vector is in top k predictions
            true_risk_vector = batch[3].unsqueeze(1).cpu()
            matches = (topk_risk_vectors == true_risk_vector).all(dim=2) # (batch_sz, k)
            acc = matches.any(dim=1).float() # (batch_sz)

            # save metric
            acc_at_k[f"acc_at_{k}"] = acc

        acc_at_k = self.add_split_prefix(acc_at_k, split)

        return acc_at_k