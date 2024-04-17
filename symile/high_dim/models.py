from argparse import Namespace
import json

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, XLMRobertaModel

from symile.losses import clip, symile, zeroshot_retrieval_logits


class AudioEncoder(nn.Module):
    def __init__(self, model_id, d, enc_hidden_size):
        super().__init__()
        self.fc = nn.Linear(enc_hidden_size, d, bias=True)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        """
        Args:
            input_features (torch.Tensor): shape (batch_sz, 80, 3000)
            attention_mask (torch.Tensor): shape (batch_sz, 3000)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.fc(x)
        x = self.layer_norm(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, d, enc_hidden_size):
        super().__init__()
        self.fc = nn.Linear(enc_hidden_size, d, bias=True)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        """
        Args:
            pixel_values (torch.Tensor): shape (batch_sz, 3, 224, 224)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.fc(x)
        x = self.layer_norm(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_id, d, enc_hidden_size):
        super().__init__()
        if model_id == "bert-base-multilingual-cased":
            self.encoder = BertModel.from_pretrained(model_id)
        elif model_id == "xlm-roberta-base" or model_id == "xlm-roberta-large":
            self.encoder = XLMRobertaModel.from_pretrained(model_id)

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

        self.fc = nn.Linear(enc_hidden_size, d, bias=True)
        self.layer_norm = nn.LayerNorm(d)

    def forward(self, x):
        """
        If not using precomputed tensors:
            Args:
                x (dict): keys are "input_ids" and "attention_mask":
                    input_ids (torch.Tensor): shape (batch_sz, len_longest_seq)
                    attention_mask (torch.Tensor): shape (batch_sz, len_longest_seq)
            Returns:
                x (torch.Tensor): shape (batch_sz, d)
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


class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)
        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        metadata = json.load(open(self.args.data_dir / self.args.metadata_filename))

        self.audio_encoder = AudioEncoder(self.args.audio_model_id, self.args.d, metadata["audio_enc_hidden_sz"])
        self.image_encoder = ImageEncoder(self.args.d, metadata["image_enc_hidden_sz"])
        self.text_encoder = TextEncoder(self.args.text_model_id, self.args.d, metadata["text_enc_hidden_sz"])

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if hasattr(self.args, "freeze_logit_scale") and self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

    def forward(self, x):
        r_a = self.audio_encoder(x["audio"])
        r_i = self.image_encoder(x["image"])
        r_t = self.text_encoder(x["text"])
        return r_a, r_i, r_t, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def training_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch["image"]))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp, "log_n": log_n},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp, self.args.efficient_loss)

        acc = self.zeroshot_retrieval_accuracy(r_a, r_t, batch, "val")

        self.log_dict({"val_loss": loss, "val_accuracy": acc},
                      on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """
        The zeroshot task is to predict which r_i corresponds to a given r_a, r_t.
        """
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        acc = self.zeroshot_retrieval_accuracy(r_a, r_t, batch, "test")

        self.log("test_accuracy", acc, sync_dist=True, prog_bar=True)

        return acc

    def on_validation_epoch_start(self):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        self.save_candidate_image_representations("val")

    def on_test_epoch_start(self):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        self.save_candidate_image_representations("test")

    def save_candidate_image_representations(self, split):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        r_i_list = []
        cls_id_list = []

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
            r_i_list.append(self.image_encoder(x["image"].to(self.device)))
            cls_id_list.append(x["cls_id"])

        # save reps
        if split == "val":
            self.r_i_val = torch.cat(r_i_list)
            self.r_i_cls_id_val = torch.cat(cls_id_list).to(self.device)
        elif split == "test":
            self.r_i_test = torch.cat(r_i_list)
            self.r_i_cls_id_test = torch.cat(cls_id_list).to(self.device)

    def zeroshot_retrieval_accuracy(self, r_a, r_t, batch, split):
        if split == "val":
            r_i = self.r_i_val
            r_i_cls_id = self.r_i_cls_id_val
        elif split == "test":
            r_i = self.r_i_test
            r_i_cls_id = self.r_i_cls_id_test

        logits = zeroshot_retrieval_logits(r_a, r_t, r_i, self.logit_scale.exp(),
                                           self.args.loss_fn)

        # pred_idx is a tensor of length batch_sz where each element is the
        # index of the r_i (across the whole test set) that maximizes the score.
        pred_idx = torch.argmax(logits, dim=1)

        # for each index in pred_idx, we get the class id (label) that corresponds
        # to the r_i at that index; so pred is a tensor of length batch_sz where
        # each element is the predicted label
        pred = r_i_cls_id[pred_idx]

        y = batch["cls_id"]

        acc = (torch.sum(y == pred) / len(y)).item()

        return acc