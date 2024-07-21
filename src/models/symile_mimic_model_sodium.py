from argparse import Namespace
import json

import lightning.pytorch as pl
import numpy as np
from rtdl_revisiting_models import FTTransformer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from datasets_sodium import SymileMIMICRetrievalDataset
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
        self.fc1 = nn.Linear(2, 256)
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
        # acc = self.zeroshot_retrieval("val_retrieval")
        acc, retrieval_acc_cxr, retrieval_acc_ecg, retrieval_acc_labs = self.zeroshot_retrieval("val_retrieval")

        self.log("val_acc", acc, sync_dist=True, prog_bar=False)

        self.log("val_acc_cxr", retrieval_acc_cxr, sync_dist=True, prog_bar=False)

        self.log("val_acc_ecg", retrieval_acc_ecg, sync_dist=True, prog_bar=False)

        self.log("val_acc_labs", retrieval_acc_labs, sync_dist=True, prog_bar=False)

        val_metrics = {
            "epoch": self.current_epoch,
            "val_loss": self.trainer.logged_metrics["val_loss_epoch"].item(),
            "val_acc": acc
        }

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

    def get_retrieval_dataset(self, split):
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
        retrieval_ds = SymileMIMICRetrievalDataset(self.args, split)

        r_c = []
        r_e = []
        r_l = []
        hadm_id = []
        label_hadm_id = []
        label = []

        # setting generator manually so that PyTorch uses it for _base_seed creation
        # (avoids altering global seed; helps ensure reproducibility)
        # (see https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4)
        for batch in DataLoader(retrieval_ds, batch_size=self.args.batch_sz_val,
                                shuffle=False, generator=torch.Generator()):
            r_c.append(self.cxr_encoder(batch["cxr"].to(self.device)))

            r_e.append(self.ecg_encoder(batch["ecg"].to(self.device)))

            if self.args.labs_model == "ftt":
                r_l.append(self.labs_encoder(batch["labs_percentiles"].to(self.device),
                                            batch["labs_missingness"].to(self.device)))
            else:
                labs = torch.cat([batch["labs_percentiles"], batch["labs_missingness"]], dim=1)
                r_l.append(self.labs_encoder(labs.to(self.device)))

            hadm_id.append(batch["hadm_id"])
            label_hadm_id.append(batch["label_hadm_id"])
            label.append(batch["label"])

        r_c = torch.cat(r_c, dim=0)
        r_e = torch.cat(r_e, dim=0)
        r_l = torch.cat(r_l, dim=0)

        assert len(r_c) == len(r_e) == len(r_l) == len(retrieval_ds), \
            "r_c, r_e, r_l, and retrieval_ds should have the same length"

        return {"r_c": r_c, "r_e": r_e, "r_l": r_l,
                "hadm_id": torch.cat(hadm_id, dim=0),
                "label_hadm_id": torch.cat(label_hadm_id, dim=0),
                "label": torch.cat(label, dim=0)}

    def get_clip_logits(self, r_x, r_y, logit_scale_exp):
        """
        assumes that r_y is the modality to predict
        """
        return logit_scale_exp * (r_y @ r_x)

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
        retrieval_ds = self.get_retrieval_dataset(split)

        # get query data (where retrieval_ds["label"] is 1)
        mask = retrieval_ds["label"] == 1
        query_r_e = retrieval_ds["r_e"][mask]
        query_r_l = retrieval_ds["r_l"][mask]
        query_hadm_id = retrieval_ds["hadm_id"][mask]
        query_label_hadm_id = retrieval_ds["label_hadm_id"][mask]

        correct_pred = 0

        print_warning = False

        # BEGIN
        query_r_c = retrieval_ds["r_c"][mask]

        correct_pred_ecg = 0
        correct_pred_labs = 0
        correct_pred_cxr = 0
        # END

        for ix, label_hadm_id in enumerate(query_label_hadm_id):
            r_e = query_r_e[ix] # (d,)
            r_l = query_r_l[ix] # (d,)
            true_hadm_id = query_hadm_id[ix]
            assert true_hadm_id == label_hadm_id, "true_hadm_id should be equal to label_hadm_id"

            # find all candidates for current label_hadm_id
            mask = retrieval_ds["label_hadm_id"] == label_hadm_id
            r_c = retrieval_ds["r_c"][mask] # (candidate_n, d)
            candidate_hadm_id = retrieval_ds["hadm_id"][mask] # (candidate_n,)

            logits = zeroshot_retrieval_logits(r_e, r_l, r_c, self.logit_scale.exp(),
                                               self.args.loss_fn).cpu()

            if logits.dim() == 1:
                logits = logits.unsqueeze(0) # ensure shape is (1, candidate_n)

            max_value = torch.max(logits)

            # find all indices that have this maximum value
            max_indices = (logits == max_value).nonzero(as_tuple=True)[1]

            if len(max_indices) > 1:
                print_warning = True

            # randomly select one of these indices (note: must use np.random.choice
            # instead of torch.randint to avoid altering the global random seed)
            pred_ix = max_indices[np.random.choice(len(max_indices))].item()

            if candidate_hadm_id[pred_ix] == true_hadm_id:
                correct_pred += 1

            # BEGIN
            r_c_q = query_r_c[ix]
            logits_cxr = self.get_clip_logits(r_c_q, r_c, self.logit_scale.exp())
            logits_cxr = logits_cxr.unsqueeze(0)
            max_value = torch.max(logits_cxr)
            max_indices = (logits_cxr == max_value).nonzero(as_tuple=True)[1]
            pred_ix = max_indices[np.random.choice(len(max_indices))].item()
            if candidate_hadm_id[pred_ix] == true_hadm_id:
                correct_pred_cxr += 1

            logits_ecg = self.get_clip_logits(r_e, r_c, self.logit_scale.exp())
            logits_ecg = logits_ecg.unsqueeze(0)
            max_value = torch.max(logits_ecg)
            max_indices = (logits_ecg == max_value).nonzero(as_tuple=True)[1]
            pred_ix = max_indices[np.random.choice(len(max_indices))].item()
            if candidate_hadm_id[pred_ix] == true_hadm_id:
                correct_pred_ecg += 1

            logits_labs = self.get_clip_logits(r_l, r_c, self.logit_scale.exp())
            logits_labs = logits_labs.unsqueeze(0)
            max_value = torch.max(logits_labs)
            max_indices = (logits_labs == max_value).nonzero(as_tuple=True)[1]
            pred_ix = max_indices[np.random.choice(len(max_indices))].item()
            if candidate_hadm_id[pred_ix] == true_hadm_id:
                correct_pred_labs += 1

            # END

        retrieval_acc = correct_pred / len(query_label_hadm_id)

        # BEGIN
        retrieval_acc_cxr = correct_pred_cxr / len(query_label_hadm_id)
        retrieval_acc_ecg = correct_pred_ecg / len(query_label_hadm_id)
        retrieval_acc_labs = correct_pred_labs / len(query_label_hadm_id)
        # END

        if print_warning:
            print("\nWARNING: Multiple indices with max value. Random index selected.\n")

        # return retrieval_acc
        return (retrieval_acc, retrieval_acc_cxr, retrieval_acc_ecg, retrieval_acc_labs)