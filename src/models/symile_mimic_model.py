from argparse import Namespace
import json

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from datasets import SymileMIMICRetrievalDataset
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
        Forward pass through the SymileMIMICModel. `x` is a list representing
        the training or validation dataset.

        Args:
            x (list): A list of length 5 with the following elements:
                - cxr (torch.Tensor): CXR training data (batch_sz, 3, 320, 320).
                - ecg (torch.Tensor): ECG training data (batch_sz, 1, 5000, 12).
                - labs_percentiles (torch.Tensor): laboratory percentiles training data (batch_sz, 50).
                - labs_missingness (torch.Tensor): missingness in laboratory training data (batch_sz, 50).
                - hadm_id (torch.Tensor): unique hospital admission ids for the training data (batch_sz,).
        """
        r_c = self.cxr_encoder(x[0])

        r_e = self.ecg_encoder(x[1])

        labs = torch.cat([x[2], x[3]], dim=1)
        r_l = self.labs_encoder(labs)

        return r_c, r_e, r_l, self.logit_scale.exp()

    def configure_optimizers(self):
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
        Calculates and logs zeroshot retrieval accuracy for the validation set,
        and updates the `run_info` dictionary with the current epoch's metrics.
        """
        acc = self.zeroshot_retrieval("val_retrieval")

        self.log("val_acc", acc, sync_dist=True, prog_bar=False)

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
        Retrieves and encodes the evaluation data (queries and candidates) for
        the specified dataset split. Each sample in the dataset is either a
        positive or a negative candidate (according to its `label`). All positive
        candidates serve as queries. Therefore the total size of the evaluation
        set is evaluation_n = num_queries * num_candidates.

        Args:
            split (str): The dataset split to evaluate ('val' or 'test').

        Returns:
            dict: A dictionary containing the encoded query data with the following keys:
                - "r_c" (torch.Tensor): Encoded representations of the CXR data (evaluation_n, d).
                - "r_e" (torch.Tensor): Encoded representations of the ECG data (evaluation_n, d).
                - "r_l" (torch.Tensor): Encoded representations of the laboratory test data (evaluation_n, d).
                - "hadm_id" (torch.Tensor): Tensor containing the hospital admission ID for each sample (evaluation_n,).
                - "label_hadm_id" (torch.Tensor): Hospital admission ID indicating the true corresponding CXR for which
                        this sample is a candidate (evaluation_n,). For positive candidates, `hamd_id` = `label_hadm_id`.
                - "label" (torch.Tensor): Tensor containing the label (1 or 0) to indicate whether the sample is a
                        positive or negative candidate (evaluation_n,).
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

            labs = torch.cat([batch["labs_percentiles"], batch["labs_missingness"]], dim=1)
            r_l.append(self.labs_encoder(labs.to(self.device)))

            hadm_id.append(batch["hadm_id"])
            label_hadm_id.append(batch["label_hadm_id"])
            label.append(batch["label"])

        r_c = torch.cat(r_c, dim=0)
        r_e = torch.cat(r_e, dim=0)
        r_l = torch.cat(r_l, dim=0)
        hadm_id = torch.cat(hadm_id, dim=0)
        label_hadm_id = torch.cat(label_hadm_id, dim=0)
        label = torch.cat(label, dim=0)

        assert len(r_c) == len(r_e) == len(r_l) == len(retrieval_ds), \
            "r_c, r_e, r_l, and retrieval_ds should have the same length"

        return {"r_c": r_c, "r_e": r_e, "r_l": r_l, "hadm_id": hadm_id,
                "label_hadm_id": label_hadm_id, "label": label}

    def zeroshot_retrieval(self, split):
        """
        Calculates zero-shot retrieval accuracy for a given dataset split ('val'
        or 'test'), where the task is to retrieve the true corresponding CXR
        image for each query ECG and labs pair.

        Args:
            split (str): The dataset split to evaluate ('val' or 'test').

        Returns:
            retrieval_acc (float): The retrieval accuracy for the specified split.
        """
        retrieval_ds = self.get_retrieval_dataset(split)

        # get query data (positive samples)
        mask = retrieval_ds["label"] == 1
        query_r_e = retrieval_ds["r_e"][mask]
        query_r_l = retrieval_ds["r_l"][mask]
        query_hadm_id = retrieval_ds["hadm_id"][mask]

        correct_pred = 0
        print_warning = False

        # loop through each query sample
        for ix, true_hadm_id in enumerate(query_hadm_id):
            r_e = query_r_e[ix] # (d,)
            r_l = query_r_l[ix] # (d,)

            # find all candidates for this query
            mask = retrieval_ds["label_hadm_id"] == true_hadm_id
            r_c = retrieval_ds["r_c"][mask] # (candidate_n, d)
            candidate_label = retrieval_ds["label"][mask] # (candidate_n,)
            assert torch.sum(candidate_label) == 1 and torch.count_nonzero(candidate_label) == 1, \
                "candidate_label must have exactly one 1 and all other elements as 0."

            logits = zeroshot_retrieval_logits(r_e, r_l, r_c, self.logit_scale.exp(),
                                               self.args.loss_fn).cpu()
            logits = logits.unsqueeze(0)

            # find all indices with the maximum value; if multiple indices have
            # the same max value, randomly select one of them (note: must use
            # np.random.choice instead of torch.randint to avoid altering the global random seed)
            max_value = torch.max(logits)
            max_indices = (logits == max_value).nonzero(as_tuple=True)[1]

            if len(max_indices) > 1:
                print_warning = True

            pred_ix = max_indices[np.random.choice(len(max_indices))].item()
            true_ix = torch.nonzero(candidate_label, as_tuple=True)[0].item()

            if pred_ix == true_ix:
                correct_pred += 1

        retrieval_acc = correct_pred / len(query_hadm_id)

        if print_warning:
            print("\nWARNING: Multiple indices with max value. Random index selected.\n")

        return retrieval_acc