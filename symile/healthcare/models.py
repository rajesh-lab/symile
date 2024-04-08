from argparse import Namespace
import json

import lightning.pytorch as pl
import numpy as np
from rtdl_revisiting_models import FTTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from datasets import EvaluationDataset
from src.losses import clip, symile
from src.healthcare.constants import CHEXPERT_LABELS


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


class LabsEncoderFTT(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.model = FTTransformer(
            n_cont_features=50,
            cat_cardinalities=[2] * 50,
            d_out=d,
            n_blocks=3,
            d_block=192,
            attention_n_heads=8,
            attention_dropout=0.2,
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=0.1,
            residual_dropout=0.0,
        )

        self.layer_norm = nn.LayerNorm(d)

    def forward(self, values, missingness):
        """
        Args:
            input_features (torch.Tensor): shape (batch_sz, 80, 3000)
            attention_mask (torch.Tensor): shape (batch_sz, 3000)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.model(values, missingness)
        x = self.layer_norm(x)
        return x


class LabsEncoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc5 = nn.Linear(2048, d)
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
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.gelu(x)
        x = self.fc5(x)
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

        if self.args.labs_model == "ftt":
            self.labs_encoder = LabsEncoderFTT(self.args.d)
        else:
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
            # train or val dataset
            if self.args.labs_model == "ftt":
                r_e = self.ecg_encoder(x[1])
                r_c = self.cxr_encoder(x[0])
                r_l = self.labs_encoder(x[2], x[3])
            else:
                r_e = self.ecg_encoder(x[1])
                r_c = self.cxr_encoder(x[0])
                r_l = self.labs_encoder(x[2])
        else:
            # query or candidate dataset
            if self.args.labs_model == "ftt":
                r_e = self.ecg_encoder(x["ecg"])
                r_c = self.cxr_encoder(x["cxr"])
                r_l = self.labs_encoder(x["labs_percentiles"], x["labs_missingness"])
            else:
                r_e = self.ecg_encoder(x["ecg"])
                r_c = self.cxr_encoder(x["cxr"])
                r_l = self.labs_encoder(x["labs"])
        return r_e, r_c, r_l, self.logit_scale.exp()

    def configure_optimizers(self):
        if self.args.labs_model == "ftt":
            # to protect some parameters from weight decay (per paper). see:
            # https://github.com/yandex-research/rtdl-revisiting-models/tree/2542a25b2adbfd0e9d18ce00f75f9f64ad2c26bd/package#ft-transformer-
            # https://github.com/yandex-research/rtdl-revisiting-models/blob/2542a25b2adbfd0e9d18ce00f75f9f64ad2c26bd/package/rtdl_revisiting_models.py#L780
            param_group_less_labs_encoder = [{"params": [p for name, p in self.named_parameters() if "labs_encoder.model" not in name]}]
            param_groups = param_group_less_labs_encoder + self.labs_encoder.model.make_parameter_groups()
            return torch.optim.AdamW(param_groups, lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _shared_step(self, batch):
        r_e, r_c, r_l, logit_scale_exp = self(batch)
        return r_e, r_c, r_l, logit_scale_exp

    def training_step(self, batch, batch_idx):
        r_e, r_c, r_l, logit_scale_exp = self._shared_step(batch)

        loss = self.loss_fn(r_e, r_c, r_l, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch[0]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                    on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n", log_n,
                on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    def on_validation_epoch_start(self):
        query_ds = EvaluationDataset(self.args, "val", "query")

        i = 0
        for batch in DataLoader(query_ds, batch_size=len(query_ds), shuffle=False):
            r_e, r_c, r_l, logit_scale_exp = self._shared_step(batch)
            i += 1
        assert i == 1, "query_ds should only have one batch"

        metrics = self.get_metrics(r_e, r_l, batch, "val")
        self.log_dict(metrics, sync_dist=True, prog_bar=False)

    def validation_step(self, batch, batch_idx):
        r_e, r_c, r_l, logit_scale_exp = self._shared_step(batch)

        loss = self.loss_fn(r_e, r_c, r_l, logit_scale_exp, self.args.efficient_loss)

        # top_k_results = self.zeroshot_accuracy(r_e, r_c, r_l)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        # self.log_dict(top_k_results,
        #               on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        return loss

    def zeroshot_accuracy(self, r_e, r_c, r_l):
        if self.args.loss_fn == "symile":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ MIP(r_e[i], r_c[0], r_l[i]) ... MIP(r_e[i], r_c[n-1], r_l[i]) ]
            # where MIP is the multilinear inner product.
            logits = (r_e * r_l) @ torch.t(r_c)
        elif self.args.loss_fn == "clip":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ r_e[i]^T r_c[0] + r_c[0]^T r_l[i]   + r_e[i]^T r_l[i] ...
            #   r_e[i]^T r_c[n-1] + r_c[n-1]^T r_l[i] + r_e[i]^T r_l[i] ]
            el = torch.diagonal(r_e @ torch.t(r_l)).unsqueeze(dim=1) # (batch_sz, 1)
            logits = el + (r_e @ torch.t(r_c)) + (r_l @ torch.t(r_c))

        # logits is square because using entire val and test sets
        logits = self.logit_scale.exp() * logits

        batch_sz = len(logits)

        top_k_results = {}

        for k in [1, 5, 10, 50, 100]:
            # ensure k does not exceed batch size
            k = min(k, batch_sz)

            topk_indices = torch.topk(logits, k, dim=1).indices.cpu()

            # correct indices are along the diagonal
            correct_indices = torch.arange(batch_sz).unsqueeze(1).expand_as(topk_indices)

            # check if correct index is in top k for each row
            top_k_accuracy = (topk_indices == correct_indices).any(dim=1)
            top_k_accuracy = top_k_accuracy.float().mean().item()

            top_k_results[f'top_{k}_accuracy'] = top_k_accuracy

        return top_k_results

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

    def get_candidates(self, split):
        candidate_ds = EvaluationDataset(self.args, split, "candidate")

        r_c = []
        hadm_id = []
        label_name = []
        label_value = []

        for batch in DataLoader(candidate_ds, batch_size=256, shuffle=False):
            r_c.append(self.cxr_encoder(batch["cxr"]))
            hadm_id.append(batch["hadm_id"])
            label_name.append(batch["label_name"])
            label_value.append(batch["label_value"])

        assert len(r_c) == len(candidate_ds), "r_c and candidate_ds should have the same length"

        return {"r_c": torch.cat(r_c, dim=0),
                "hadm_id": hadm_id,
                "label_name": label_name,
                "label_value": label_value}

    def get_metrics(self, r_e_query, r_l_query, batch, split):
        candidates = self.get_candidates(split)

        precisions = {}

        precision_cxr_labels = {}
        acc_cxr_labels = {}

        for label in CHEXPERT_LABELS:
            query_indices = [i for i, l in enumerate(batch["label_name"]) if l == label]
            r_e = r_e_query[query_indices]
            r_l = r_l_query[query_indices]

            candidate_indices = [i for i, l in enumerate(self.candidates_data["label_name"]) if l == label]

            candidate_labels = [self.candidates_data["label_value"][i] for i in candidate_indices]
            candidate_labels = torch.tensor(candidate_labels).unsqueeze(0).to(self.device) #(1, n)

            candidate_cxrs = self.candidates_data["cxr"][candidate_indices]
            r_c = self.get_cxr_representations(candidate_cxrs)

            logits = get_logits(self, r_e, r_l, r_c, self.logit_scale.exp())

            # get indices of sorted scores
            _, indices_sorted = logits.sort(dim=1, descending=True)

            # sort labels based on sorted indices
            # sorted_labels has shape (batch_size, n); each row contains 1 or 0 (positive or negative)
            # for `label`, reordered to match the sorted logit scores. In other words, the highest score
            # in each row of logits corresponds to the first label in the same row of sorted_labels, etc.
            # .gather(1, indices_sorted) reorders each row of the candidate_labels_tensor to align with
            # the descending order of scores in logits.
            sorted_labels = candidate_labels.expand_as(logits).gather(1, indices_sorted)

            for k in [1, 5, 10, 50, 100]:
                top_k_labels = sorted_labels[:, :k]
                precision_at_k = (top_k_labels.sum(dim=1) / k).mean().item()
                precisions[f"{label}_precision_at_{k}"] = precision_at_k

            # calculcate mean precision across all labels
            precision_sums = defaultdict(float)
            counts = defaultdict(int)

            for key, value in precisions.items():
                _, k = key.rsplit("_at_", 1)
                precision_sums[k] += value
                counts[k] += 1

            for k in precision_sums:
                precisions[f"mean_precision_at_{k}"] = precision_sums[k] / counts[k]

        acc_true_cxr = {}

        return precisions

    def test_step(self, batch, batch_idx):
        """
        The zeroshot task is to predict which r_i corresponds to a given r_a, r_t.
        """
        r_e, r_c, r_l, logit_scale_exp = self._shared_step(batch)

        metrics = self.get_metrics(r_e, r_l, batch, "test")
        self.log_dict(metrics, sync_dist=True, prog_bar=False)