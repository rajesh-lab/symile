from argparse import Namespace
from collections import defaultdict
import json

import lightning.pytorch as pl
import numpy as np
from rtdl_revisiting_models import FTTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from datasets import SymileMIMICEvaluationDataset
from src.constants import CHEXPERT_LABELS
from src.losses import clip, symile, zeroshot_retrieval_logits
from src.utils import PathToStrEncoder


class CXREncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize the CXREncoder, which encodes chest X-ray (CXR) images using
        a modified ResNet-50 architecture.

        If `args.pretrained` is True, the ResNet-50 model is initialized with
        pre-trained weights from the ImageNet dataset ("IMAGENET1K_V2"). The
        fully connected layer (fc) of ResNet-50 is replaced with a new Linear
        layer to match the desired output dimensionality (`args.d`). A LayerNorm
        layer is added to normalize the output features.

        Args:
            args (Namespace): A namespace object containing configuration for the model.
        """
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
            x (torch.Tensor): CXR data (batch_sz, 3, 320, 320).
        Returns:
            x (torch.Tensor): learned CXR representation (batch_sz, d)
        """
        x = self.resnet(x)
        x = self.layer_norm(x)
        return x


class ECGEncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize the ECGEncoder, which encodes ECG data using a modified
        ResNet-18 architecture.

        If `args.pretrained` is True, the ResNet-18 model is initialized with
        pre-trained weights from the ImageNet dataset ("IMAGENET1K_V1"). The
        first convolutional layer of ResNet-18 is modified to accept single-
        channel input by changing the number of input channels to 1. The fully
        connected layer (fc) of ResNet-18 is replaced with a new Linear layer to
        match the desired output dimensionality (`args.d`). A LayerNorm layer is
        added to normalize the output features.

        Args:
            args (Namespace): A namespace object containing configuration for
                              the model.
        """
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
            x (torch.Tensor): ECG data (batch_sz, 1, 5000, 12).
        Returns:
            x (torch.Tensor): learned ECG representation (batch_sz, d)
        """
        x = self.resnet(x)
        x = self.layer_norm(x)
        return x


class LabsEncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize the LabsEncoder, which encodes laboratory test results using
        a multi-layer perceptron (MLP) architecture.

        The encoder consists of three fully connected layers (fc1, fc2, fc3) with
        GELU activation functions. A LayerNorm layer is added to normalize the
        output features.

        Args:
            args (Namespace): A namespace object containing configuration for the model.
        """
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, args.d)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(args.d)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): concatenated laboratory percentiles and missingness
                              data (batch_sz, 100).
        Returns:
            x (torch.Tensor): learned labs representation (batch_sz, d)
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.gelu(x)
        x = self.fc3(x)
        x = self.layer_norm(x)
        return x


class LabsEncoderFTT(nn.Module):
    def __init__(self, args):
        """
        Initialize the LabsEncoderFTT, which encodes laboratory test results
        using a FT-Transformer architecture (see
        https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/README.md
        and https://arxiv.org/pdf/2106.11959 for details). A LayerNorm layer is
        added to normalize the output features.

        Args:
            args (Namespace): A namespace object containing configuration for the model.
        """
        super().__init__()

        self.model = FTTransformer(
            n_cont_features=50,
            cat_cardinalities=[2] * 50,
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

    def forward(self, values, missingness):
        """
        FTTransformer expects continuous (laboratory percentile) and categorical
        (missingness indicator) features to be passed separately.

        Args:
            values (torch.Tensor): laboratory percentiles data (batch_sz, 50).
            missingness (torch.Tensor): missingness in laboratory data (batch_sz, 50).
        Returns:
            x (torch.Tensor): learned labs representation (batch_sz, d)
        """
        x = self.model(values, missingness)
        x = self.layer_norm(x)
        return x


class SymileMIMICModel(pl.LightningModule):
    def __init__(self, **args):
        """
        Initialize the PyTorch Lightning module, which learns CXR, ECG, and labs
        representations using either the Symile or CLIP loss.

        Args:
            **args: Arguments containing model and training configuration.
        """
        super().__init__()

        self.save_hyperparameters()

        self.args = Namespace(**args)

        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.ecg_encoder = ECGEncoder(self.args)
        self.cxr_encoder = CXREncoder(self.args)

        if self.args.labs_model == "ftt":
            self.labs_encoder = LabsEncoderFTT(self.args)
        else:
            self.labs_encoder = LabsEncoder(self.args)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        # for logging attributes and metrics
        self.run_info = {}

    def forward(self, x):
        """
        Forward pass through the SymileMIMICModel.

        If `x` is a list, it represents the training or validation dataset. If
        `x` is a dictionary, it represents the query or candidate dataset.

        Args:
            x (list): If x is a list, it has length 5 with the following elements:
                - cxr (torch.Tensor): CXR training data (batch_sz, 3, 320, 320).
                - ecg (torch.Tensor): ECG training data (batch_sz, 1, 5000, 12).
                - labs_percentiles (torch.Tensor): laboratory percentiles training data (batch_sz, 50).
                - labs_missingness (torch.Tensor): missingness in laboratory training data (batch_sz, 50).
                - hadm_id (torch.Tensor): unique hospital admission ids for the training data (batch_sz,).
                      If x is a dictionary, it has the same elements with keys
                      "cxr", "ecg", "labs_percentiles", "labs_missingness", and
                      "hadm_id", in addition to "label_name" and "label_value":
                If x is a dictionary, it has the above elements as keys, in addition to:
                - label_name (list): List of label names corresponding to each candidate (len candidate_sz).
                - label_value (torch.Tensor): Binary indicator Tensor where 1 indicates that the candidate
                                                is positive for the label and 0 indicates that it is negative
                                                for the label (candidate_sz,).
        """
        # training or validation dataset
        if isinstance(x, list):
            r_c = self.cxr_encoder(x[0])

            r_e = self.ecg_encoder(x[1])

            if self.args.labs_model == "ftt":
                # FTTransformer expects continuous (laboratory percentile) and
                # categorical (missingness indicator) features to be passed separately.
                r_l = self.labs_encoder(x[2], x[3])
            else:
                labs = torch.cat([x[2], x[3]], dim=1)
                r_l = self.labs_encoder(labs)
        # query or candidate dataset
        else:
            breakpoint()
            r_c = self.cxr_encoder(x["cxr"].to(self.device))

            r_e = self.ecg_encoder(x["ecg"].to(self.device))

            if self.args.labs_model == "ftt":
                r_l = self.labs_encoder(x["labs_percentiles"].to(self.device),
                                        x["labs_missingness"].to(self.device))
            else:
                labs = torch.cat([x["labs_percentiles"], x["labs_missingness"]], dim=1)
                r_l = self.labs_encoder(labs.to(self.device))

        return r_c, r_e, r_l, self.logit_scale.exp()

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

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch (list): A list of length 5 representing the training batch with elements:
                - cxr (torch.Tensor): CXR data (batch_sz, 3, 320, 320).
                - ecg (torch.Tensor): ECG data (batch_sz, 1, 5000, 12).
                - labs_percentiles (torch.Tensor): laboratory percentiles data (batch_sz, 50).
                - labs_missingness (torch.Tensor): missingness in laboratory data (batch_sz, 50).
                - hadm_id (torch.Tensor): unique hospital admission ids for the data (batch_sz,).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        r_c, r_e, r_l, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_e, r_l, logit_scale_exp, self.args.negative_sampling)

        # tracking to help evaluate optimization (given total correlation lower bound established in paper)
        log_n = np.log(len(batch[0]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp, "log_n": log_n},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Args:
            batch (list): A list of length 5 representing the validation batch.
                          Refer to the `training_step` method for detailed
                          descriptions of the elements and their shapes.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        r_c, r_e, r_l, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_c, r_e, r_l, logit_scale_exp, self.args.negative_sampling)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        """
        Calculates and logs zeroshot retrieval metrics for the validation set,
        and updates the `run_info` dictionary with the current epoch's metrics.
        """
        metrics = self.zeroshot_retrieval("val")

        self.log_dict(metrics, sync_dist=True, prog_bar=False)

        val_metrics = {
            "epoch": self.current_epoch,
            "val_loss": self.trainer.logged_metrics["val_loss_epoch"].item()
        }

        val_metrics.update({key: value for key, value in metrics.items()})

        self.run_info.setdefault("validation_metrics", []).append(val_metrics)

    def on_train_end(self):
        """
        Stores the arguments and logging information in the `run_info` attribute,
        which is then saved to a JSON file in the specified directory.
        """
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
        breakpoint()
        metrics = self.zeroshot_retrieval("test")

        self.log_dict(metrics, sync_dist=True, prog_bar=False)

    def get_queries(self, split):
        """
        Retrieves and encodes the query data for the specified dataset split.

        Args:
            split (str): The dataset split to evaluate ('val' or 'test').

        Returns:
            dict: A dictionary containing the encoded query data with the following keys:
                - "r_e" (torch.Tensor): Encoded representations of the ECG data (query_sz, d).
                - "r_l" (torch.Tensor): Encoded representations of the laboratory test data (query_sz, d).
                - "hadm_id" (torch.Tensor): Tensor containing the hospital admission IDs (query_sz,).
                - "label_name" (list): List of label names corresponding to each query (len query_sz).
        """
        query_ds = SymileMIMICEvaluationDataset(self.args, split, "query")

        r_e = []
        r_l = []
        hadm_id = []
        label_name = []

        for batch in DataLoader(query_ds, batch_size=len(query_ds), shuffle=False):
            r_e.append(self.ecg_encoder(batch["ecg"].to(self.device)))

            if self.args.labs_model == "ftt":
                r_l.append(self.labs_encoder(batch["labs_percentiles"].to(self.device),
                                             batch["labs_missingness"].to(self.device)))
            else:
                labs = torch.cat([batch["labs_percentiles"], batch["labs_missingness"]], dim=1)
                r_l.append(self.labs_encoder(labs.to(self.device)))

            hadm_id.append(batch["hadm_id"])

            label_name += batch["label_name"]

        r_e = torch.cat(r_e, dim=0)
        r_l = torch.cat(r_l, dim=0)

        assert len(r_e) == len(r_l) == len(query_ds), "r_e, r_l, and query_ds should have the same length"

        return {"r_e": r_e, "r_l": r_l, "hadm_id": torch.cat(hadm_id, dim=0),
                "label_name": label_name}

    def get_candidates(self, split):
        """
        Retrieves and encodes the candidate data for the specified dataset split.

        Args:
            split (str): The dataset split to evaluate ('val' or 'test').

        Returns:
            dict: A dictionary containing the encoded candidate data with the following keys:
                - "r_c" (torch.Tensor): Encoded representations of the CXR data (candidate_sz, d).
                - "hadm_id" (torch.Tensor): Unique hospital admission IDs (candidate_sz,).
                - "label_name" (list): List of label names corresponding to each candidate (len candidate_sz).
                - "label_value" (torch.Tensor): Binary indicator Tensor where 1 indicates that the candidate
                                                is positive for the label and 0 indicates that it is negative
                                                for the label (candidate_sz,) (since each label has positive
                                                and negative candidates).
        """
        candidate_ds = SymileMIMICEvaluationDataset(self.args, split, "candidate")

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

    def get_mean_metric_across_labels(self, metric_dict, metric_name):
        """
        Calculates the mean metrics across all labels for different values of k.

        Args:
            metric_dict (dict): A dictionary containing metrics for different labels and k values.
                                The keys are in the format "<label>_<metric_name>_at_<k>"
                                (e.g., "Atelectasis_label_retrieval_acc_at_1").
            metric_name (str): The base name of the metric to calculate mean values for
                               (e.g., "label_retrieval_acc").

        Returns:
            dict: A dictionary containing the mean metrics for each k value, with keys in the format
                "<metric_name>_at_<k>".
        """
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
        """
        Adds a prefix to all metric names (keys) in the given dictionary to
        indicate the dataset split.

        Args:
            metrics (dict): A dictionary containing metric names as keys (e.g.
                            "label_retrieval_acc_at_5").
            split (str): The dataset split to add as a prefix ("val" or "test")

        Returns:
            dict: A new dictionary with the same metric values, but with each
                  metric name is prefixed with the specified split (e.g.
                  "label_retrieval_acc_at_1" -> "val_label_retrieval_acc_at_1").
        """
        return {f"{split}_{key}": value for key, value in metrics.items()}

    def zeroshot_retrieval(self, split):
        """
        Calculates zero-shot retrieval metrics for a given dataset split ('val'
        or 'test'). For both tasks, ECG and labs are used to predict the most
        probable corresponding CXR. "CXR image retrieval" task is to retrieve
        the true corresponding CXR image for each query ECG and labs pair. "CXR
        label retrieval" task is to retrieve the pathology label of the true
        corresponding CXR image for each query ECG and labs pair.

        For each label in CHEXPERT_LABELS, the function computes the logits for
        the retrieval task, and then calculates label retrieval precision at k,
        label retrieval accuracy at k, and image retrieval accuracy at k for k
        values of 1, 5, and 10. The calculated metrics are saved in dictionaries
        and averaged across all labels. The function returns a dictionary of
        metrics with a prefix indicating the dataset split.

        Args:
            split (str): The dataset split to evaluate ('val' or 'test').

        Returns:
            dict: A dictionary containing various zero-shot retrieval accuracy metrics for each label,
                including precision at k, accuracy at k, and true CXR retrieval accuracy at k.
        """
        # retrieve the query and candidate sets for the specified split
        queries = self.get_queries(split)
        candidates = self.get_candidates(split)

        label_retrieval_precision_at_k_dict = {}
        label_retrieval_acc_at_k_dict = {}
        img_retrieval_acc_at_k_dict = {}

        for label in CHEXPERT_LABELS:
            # get query set for `label`
            label_mask_q = torch.tensor([name == label for name in queries["label_name"]])
            r_e = queries["r_e"][label_mask_q]
            r_l = queries["r_l"][label_mask_q]
            true_hadm_ids = queries["hadm_id"][label_mask_q]

            # get candidate set for `label`
            label_mask_c = torch.tensor([name == label for name in candidates["label_name"]])
            r_c = candidates["r_c"][label_mask_c]
            r_c_labels = candidates["label_value"][label_mask_c]
            r_c_hadm_id = candidates["hadm_id"][label_mask_c]

            # logits are (query_sz, candidate_sz) or (5, 125)
            logits = zeroshot_retrieval_logits(r_e, r_l, r_c, self.logit_scale.exp(),
                                               self.args.loss_fn).cpu()

            for k in [1, 5, 10]:
                # get indices of top k logits
                _, topk_indices = torch.topk(logits, k, dim=1) # (query_sz, k)

                ### LABEL RETRIEVAL ###

                # get positive/negative labels at those indices
                topk_labels = r_c_labels[topk_indices] # (query_sz, k)

                # calculate metrics
                precision_at_k = topk_labels.sum(dim=1) / k
                mean_precision_at_k = precision_at_k.mean().item()

                acc_at_k = torch.any(topk_labels, dim=1).float()
                mean_acc_at_k = torch.mean(acc_at_k).item()

                # save metrics
                label_retrieval_precision_at_k_dict[f"{label}_label_retrieval_precision_at_{k}"] = mean_precision_at_k
                label_retrieval_acc_at_k_dict[f"{label}_label_retrieval_acc_at_{k}"] = mean_acc_at_k

                ### CXR IMAGE RETRIEVAL ###

                # map indices to hadm_ids
                topk_hadm_ids = r_c_hadm_id[topk_indices] # (query_sz, k)

                # check if true hadm_id is in top k predicted hadm_ids
                img_retrieval_acc_at_k = (true_hadm_ids.unsqueeze(1) == topk_hadm_ids).any(dim=1).float()
                mean_img_retrieval_acc_at_k = torch.mean(img_retrieval_acc_at_k).item()

                # save metric
                img_retrieval_acc_at_k_dict[f"{label}_img_retrieval_acc_at_{k}"] = mean_img_retrieval_acc_at_k


        label_retrieval_precision_at_k_mean = self.get_mean_metric_across_labels(label_retrieval_precision_at_k_dict, "label_retrieval_precision")
        label_retrieval_acc_at_k_mean = self.get_mean_metric_across_labels(label_retrieval_acc_at_k_dict, "label_retrieval_acc")
        img_retrieval_acc_at_k_mean = self.get_mean_metric_across_labels(img_retrieval_acc_at_k_dict, "img_retrieval_acc")

        metrics = label_retrieval_precision_at_k_dict | label_retrieval_acc_at_k_dict | img_retrieval_acc_at_k_dict | \
                  label_retrieval_precision_at_k_mean | label_retrieval_acc_at_k_mean | img_retrieval_acc_at_k_mean
        metrics = self.add_split_prefix(metrics, split)

        return metrics