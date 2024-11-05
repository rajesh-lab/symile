"""Models for the Symile-M3 experiments."""
from argparse import Namespace
import json

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel

from losses import clip, symile, zeroshot_retrieval_logits
from utils import PathToStrEncoder


class AudioEncoder(nn.Module):
    def __init__(self, args, enc_hidden_size):
        """
        Initialize the AudioEncoder, which processes precomputed audio
        representations from a separate pretrained audio encoder. The AudioEncoder
        applies a linear layer and layer normalization to these representations.

        If missingness is specified in the arguments, two embeddings are learned:
        one for observed data points and one for missing data points. For observed
        data, concatenates its encoder representation with the observed embedding,
        and pass the combined vector into the linear projection head. For missing
        data, concatenates the mean encoder representation from observed training
        samples with the missing embedding, and pass the combined vector into the
        linear projection head.

        Args:
            args (Namespace): Arguments containing the configuration for the model.
            enc_hidden_size (int): Size of audio encoder's last hidden layer.
        """
        super().__init__()

        self.args = args

        if getattr(args, "missingness", False):
            self.missingness_embed = nn.Embedding(2, enc_hidden_size)
            self.fc = nn.Linear(enc_hidden_size*2, args.d, bias=True)
        else:
            self.fc = nn.Linear(enc_hidden_size, args.d, bias=True)

        self.layer_norm = nn.LayerNorm(args.d)

    def forward(self, audio_embed, missingness_ind):
        """
        Args:
            audio_embed (torch.Tensor): precomputed audio representations (batch_sz, 1280).
            missingness_ind (torch.Tensor): binary indicators, where 0 indicates the audio
                data is observed and 1 indicates the audio data is missing (batch_sz).
        Returns:
            x (torch.Tensor): learned audio representations (batch_sz, d)
        """
        if getattr(self.args, "missingness", False):
            missingness_embed = self.missingness_embed(missingness_ind)
            x = torch.cat((audio_embed, missingness_embed), dim=1)
            x = self.fc(x)
        else:
            x = self.fc(audio_embed)

        x = self.layer_norm(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, args, enc_hidden_size):
        """
        Initialize the ImageEncoder, which processes precomputed image
        representations from a separate pretrained image encoder. The ImageEncoder
        applies a linear layer and layer normalization to these representations.

        If missingness is specified in the arguments, two embeddings are learned:
        one for observed data points and one for missing data points. For observed
        data, concatenates its encoder representation with the observed embedding,
        and pass the combined vector into the linear projection head. For missing
        data, concatenates the mean encoder representation from observed training
        samples with the missing embedding, and pass the combined vector into the
        linear projection head.

        Args:
            args (Namespace): Arguments containing the configuration for the model.
            enc_hidden_size (int): Size of image encoder's last hidden layer.
        """
        super().__init__()

        self.args = args

        if getattr(args, "missingness", False):
            self.missingness_embed = nn.Embedding(2, enc_hidden_size)
            self.fc = nn.Linear(enc_hidden_size*2, args.d, bias=True)
        else:
            self.fc = nn.Linear(enc_hidden_size, args.d, bias=True)

        self.layer_norm = nn.LayerNorm(args.d)

    def forward(self, image_embed, missingness_ind):
        """
        Args:
            image_embed (torch.Tensor): precomputed image representations (batch_sz, 1024).
            missingness_ind (torch.Tensor): binary indicators, where 0 indicates the image
                data is observed and 1 indicates the image data is missing (batch_sz).
        Returns:
            x (torch.Tensor): learned image representations (batch_sz, d)
        """
        if getattr(self.args, "missingness", False):
            missingness_embed = self.missingness_embed(missingness_ind)
            x = torch.cat((image_embed, missingness_embed), dim=1)
            x = self.fc(x)
        else:
            x = self.fc(image_embed)

        x = self.layer_norm(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, args, enc_hidden_size):
        """
        Initialize the TextEncoder, which freezes all parameters except for those
        in model's embedding layer and first encoder layer, which are fine-tuned.
        A linear layer and layer normalization are applied to the encoded features.

        If missingness is specified in the arguments, resizes the token embeddings
        to accommodate the additional token for missing text data.

        Args:
            args (Namespace): Arguments containing the configuration for the model.
            enc_hidden_size (int): Size of the hidden layer for encoding text features.
        """
        super().__init__()

        self.encoder = AutoModel.from_pretrained(args.text_model_id)

        if getattr(args, "missingness", False):
            self.encoder.resize_token_embeddings(args.tokenizer_len)

        self.embeddings = self.encoder.embeddings
        self.encoder_layer = self.encoder.encoder.layer[0]

        # first freeze all parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

        # then unfreeze relevant parameters
        for p in self.embeddings.parameters():
            p.requires_grad = True
        for p in self.encoder_layer.parameters():
            p.requires_grad = True

        self.fc = nn.Linear(enc_hidden_size, args.d, bias=True)
        self.layer_norm = nn.LayerNorm(args.d)

    def forward(self, x):
        """
        Args:
            x (dict): A dictionary containing the following key-value pairs:
                - input_ids (torch.Tensor): token_ids for the text (batch_sz, max_token_len).
                - attention_mask (torch.Tensor): attention mask for the text (batch_sz, max_token_len).

        Returns:
            x (torch.Tensor): learned text representations (batch_sz, d).
        """
        # https://github.com/huggingface/transformers/blob/a0857740c0e6127485c11476650314df3accc2b6/src/transformers/modeling_utils.py#L941
        # attention mask has shape (batch_sz, seq_len)
        # we make the mask broadcastable to (batch_sz, num_heads, seq_len, seq_len)
        extended_attention_mask = x["attention_mask"][:, None, None, :]
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.encoder.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.encoder.dtype).min

        embedding_output = self.embeddings(x["input_ids"])
        encoder_outputs = self.encoder_layer(embedding_output, attention_mask=extended_attention_mask)
        x = encoder_outputs[0]
        x = self.fc(x)
        x = x.mean(dim=1)
        x = self.layer_norm(x)
        return x


class SymileM3Model(pl.LightningModule):
    def __init__(self, **args):
        """
        Initialize the PyTorch Lightning module, which learns audio, image, and
        text representations using either the Symile or CLIP loss.

        Args:
            **args: Arguments containing model and training configuration.
        """
        super().__init__()

        self.save_hyperparameters()

        self.args = Namespace(**args)

        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        try:
            metadata = json.load(open(self.args.data_dir / self.args.metadata_filename))
        except (FileNotFoundError, AttributeError):
            metadata = json.load(open(self.args.data_dir / self.args.metadata_pt))

        self.audio_encoder = AudioEncoder(self.args, metadata["audio_enc_hidden_sz"])
        self.image_encoder = ImageEncoder(self.args, metadata["image_enc_hidden_sz"])
        self.text_encoder = TextEncoder(self.args, metadata["text_enc_hidden_sz"])

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        if self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        # for logging attributes and metrics
        self.run_info = {}
        self.val_step_accuracies = []
        self.test_step_accuracies = []

        # used during testing if saving representations
        self.r_a_test_save = torch.empty(0)
        self.r_i_test_save = torch.empty(0)
        self.r_t_test_save = torch.empty(0)

    def forward(self, x):
        """
        Forward pass through the Symile-M3 model.

        Args:
            x (dict): A dictionary containing the following key-value pairs:
                - text (dict): Dictionary with keys "input_ids" and "attention_mask" whose values are each (batch_sz, max_token_len).
                - image (torch.Tensor): Tensor with image data (batch_sz, 1024).
                - audio (torch.Tensor): Tensor with audio data (batch_sz, 1280).
                - cls_id (torch.float32): Tensor containing the class id for the sample (as determined by the image class name) (batch_sz).
                - idx (torch.float32): Tensor containing the unique identifier for the sample (batch_sz).
                - lang_id (int): Integer representing the language id for the sample (batch_sz).
                - text_missing (torch.int32): Integer indicating whether the text data is observed (0) or missing (1) (batch_sz).
                - image_missing (torch.int32): Integer indicating whether the image data is observed (0) or missing (1) (batch_sz).
                - audio_missing (torch.int32): Integer indicating whether the audio data is observed (0) or missing (1) (batch_sz).
                - all_observed (int): Integer indicating whether all modalities are observed (1) or if some modalities are missing (0) (batch_sz).

        Returns:
            A tuple containing:
                - r_a (torch.Tensor): Encoded audio representations.
                - r_i (torch.Tensor): Encoded image representations.
                - r_t (torch.Tensor): Encoded text representations.
                - logit_scale_exp (torch.Tensor): Exponentiated logit scale.
        """
        r_a = self.audio_encoder(x["audio"], x["audio_missing"])
        r_i = self.image_encoder(x["image"], x["image_missing"])
        r_t = self.text_encoder(x["text"])
        return r_a, r_i, r_t, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch (dict): A dictionary containing the input batch. Refer to the
                `forward` method for detailed descriptions of the keys and their shapes.
            batch_idx (int): Index of the batch.

        Returns:
            (torch.Tensor): The computed loss for the batch.
        """
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp, self.args.negative_sampling)

        # tracking to help evaluate optimization (given total correlation lower bound established in paper)
        log_n = np.log(len(batch["image"]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp, "log_n": log_n},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        The zeroshot retrieval task is to predict image given audio and text.

        Args:
            batch (dict): A dictionary containing the input batch. Refer to the
                `forward` method for detailed descriptions of the keys and their shapes.
            batch_idx (int): Index of the batch.

        Returns:
            (torch.Tensor): The computed loss for the batch.
        """
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp, self.args.negative_sampling)

        accuracies = self.zeroshot_retrieval(r_a, r_t, batch, "val")

        self.val_step_accuracies.extend(accuracies)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        The zeroshot retrieval task is to predict image given audio and text.

        Args:
            batch (dict): A dictionary containing the input batch. Refer to the
                `forward` method for detailed descriptions of the keys and their shapes.
            batch_idx (int): Index of the batch.

        Returns:
            (torch.Tensor): The computed loss for the batch.
        """
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        accuracies = self.zeroshot_retrieval(r_a, r_t, batch, "test")

        self.test_step_accuracies.extend(accuracies)

    def on_validation_epoch_start(self):
        """
        Computes all val image representations.
        """
        assert self.val_step_accuracies == [], "val_step_accuracies is not empty"

        self.save_candidate_image_representations("val")

    def on_test_epoch_start(self):
        """
        Computes all test image representations.
        """
        assert self.test_step_accuracies == [], "test_step_accuracies is not empty"

        self.save_candidate_image_representations("test")

    def on_validation_epoch_end(self):
        """
        Calculates mean validation accuracy from the recorded step accuracies,
        logs mean accuracy, and stores validation metrics for the current epoch.
        It also clears the list of validation step accuracies for the next epoch.
        """
        mean_acc = sum(self.val_step_accuracies) / len(self.val_step_accuracies)

        self.log("val_acc", mean_acc, sync_dist=True, prog_bar=True)

        val_metrics = {
            "epoch": self.current_epoch,
            "val_loss": self.trainer.logged_metrics["val_loss_epoch"].item(),
            "val_acc": mean_acc
        }

        self.run_info.setdefault("validation_metrics", []).append(val_metrics)

        self.val_step_accuracies.clear()

    def on_test_epoch_end(self):
        """
        Calculates mean test accuracy from the recorded step accuracies and logs
        mean accuracy. It also clears the list of test step accuracies for the
        next epoch. Additionally, if saving representations is enabled, it saves
        the test representations to the specified directory.
        """
        mean_acc = sum(self.test_step_accuracies) / len(self.test_step_accuracies)

        self.log("test_acc", mean_acc, sync_dist=True, prog_bar=True)

        self.test_step_accuracies.clear()

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

    def save_candidate_image_representations(self, split):
        """
        Computes all image representations for the specified dataset split.
        (either 'val' or 'test') and then saves these representations for later use.
        Note that this method is only called during validation and testing; therefore,
        during evaluation, we only look at samples where all modalities are observed.
        """
        r_i_list = []
        cls_id_list = []

        # get dataloader
        if split == "val":
            dl = self.trainer.datamodule.val_dataloader()
        elif split == "test":
            if self.trainer.datamodule is None:
                dl = getattr(self, "test_dataloader", None)
            else:
                dl = self.trainer.datamodule.test_dataloader()

        # loop through dataloader
        for x in dl:
            # only look at samples where all modalities are observed
            mask = x["all_observed"] == 1

            image = x["image"][mask].to(self.device)
            image_missing = x["image_missing"][mask].to(self.device)
            cls_id = x["cls_id"][mask]

            r_i_list.append(self.image_encoder(image, image_missing))
            cls_id_list.append(cls_id)

        # save reps
        if split == "val":
            self.r_i_val = torch.cat(r_i_list)
            self.r_i_cls_id_val = torch.cat(cls_id_list).to(self.device)
        elif split == "test":
            self.r_i_test = torch.cat(r_i_list)
            self.r_i_cls_id_test = torch.cat(cls_id_list).to(self.device)

    def zeroshot_retrieval(self, r_a, r_t, batch, split):
        """
        Perform zeroshot retrieval to predict images given audio and text representations.

        Args:
            r_a (torch.Tensor): Learned audio representations of shape (batch_sz, d).
            r_t (torch.Tensor): Learned text representations of shape (batch_sz, d).
            batch (dict): A dictionary containing the input batch. Refer to the `forward` method
                for detailed descriptions of the keys and their shapes.
            split (str): The dataset split to process ('val' or 'test').

        Returns:
            list: A list of accuracies for each sample in the batch.
        """
        # get candidate image representations and class ids
        if split == "val":
            r_i = self.r_i_val
            r_i_cls_id = self.r_i_cls_id_val
        elif split == "test":
            r_i = self.r_i_test
            r_i_cls_id = self.r_i_cls_id_test

        mask = batch["all_observed"] == 1

        if split == "test":
            assert mask.all(), "All values should be observed in test set."

        r_a = r_a[mask]
        r_t = r_t[mask]

        # logits is a tensor of shape (num_samples_all_observed, num_candidates)
        # where each element in a row is the score for the corresponding image candidate.
        logits = zeroshot_retrieval_logits(r_i, [r_a, r_t], self.logit_scale.exp(),
                                           self.args.loss_fn)

        # pred_idx is a tensor of length batch_sz where each element is the
        # index of the r_i (across the whole all-observed eval set) that maximizes the score.
        pred_idx = torch.argmax(logits, dim=1)

        # for each index in pred_idx, we get the class id (label) that corresponds
        # to the r_i at that index; so pred is a tensor of length batch_sz where
        # each element is the predicted label
        pred = r_i_cls_id[pred_idx]

        y = batch["cls_id"][mask]

        accuracies = (y == pred).float().tolist()

        return accuracies