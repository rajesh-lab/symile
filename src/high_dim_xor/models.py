from argparse import Namespace

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel, WhisperModel, XLMRobertaModel
try:
    import wandb
except ImportError:
    wandb = None

from src.losses import clip, symile
from src.utils import l2_normalize


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size, d, layer_norm_eps):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.linear_projection = nn.Linear(hidden_size, d, bias=False)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear_projection(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self, model_id, d, freeze_encoders):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(model_id).encoder

        if freeze_encoders:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.projection_head = ProjectionHead(self.encoder.config.hidden_size, d,
                                              self.encoder.layer_norm.eps)

    def forward(self, x):
        """
        Args:
            input_features (torch.Tensor): shape (batch_sz, 80, 3000)
            attention_mask (torch.Tensor): shape (batch_sz, 3000)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        if type(x) is dict: # not using precomputed tensors
            x = self.encoder(input_features=x["input_features"],
                             attention_mask=x["attention_mask"])
            x = x["last_hidden_state"]

            # select first embedding as done by ImageBind:
            # https://github.com/facebookresearch/ImageBind/blob/95d27c7fd5a8362f3527e176c3a80ae5a4d880c0/imagebind/models/imagebind_model.py#L391C3-L391C3
            x = x[:, 0, :]

        x = self.projection_head(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, model_id, d, freeze_encoders):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_id)

        if freeze_encoders:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.projection_head = ProjectionHead(self.encoder.config.hidden_size, d,
                                              self.encoder.config.layer_norm_eps)

    def forward(self, x):
        """
        Args:
            pixel_values (torch.Tensor): shape (batch_sz, 3, 224, 224)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        if type(x) is dict: # not using precomputed tensors
            x = self.encoder(pixel_values=x["pixel_values"])
            x = x["last_hidden_state"]

            # select first embedding as done by transformers' CLIP and ImageBind:
            # https://github.com/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/clip/modeling_clip.py#L894
            # https://github.com/facebookresearch/ImageBind/blob/95d27c7fd5a8362f3527e176c3a80ae5a4d880c0/imagebind/models/imagebind_model.py#L380
            x = x[:, 0, :]

        x = self.projection_head(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_id, d, freeze_encoders, feat_token_id):
        super().__init__()
        self.feat_token_id = feat_token_id
        if model_id == "bert-base-multilingual-cased":
            self.encoder = BertModel.from_pretrained(model_id)
        elif model_id == "xlm-roberta-base":
            self.encoder = XLMRobertaModel.from_pretrained(model_id)

        if freeze_encoders:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        self.projection_head = ProjectionHead(self.encoder.config.hidden_size, d,
                                              self.encoder.config.layer_norm_eps)

    def forward(self, x):
        """
        Args:
            input_ids (torch.Tensor): shape (batch_sz, len_longest_seq)
            attention_mask (torch.Tensor): shape (batch_sz, len_longest_seq)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        if type(x) is dict: # not using precomputed tensors
            x_arg = x
            x = self.encoder(input_ids=x_arg["input_ids"],
                             attention_mask=x_arg["attention_mask"])
            x = x["last_hidden_state"]

            # take features from EOS or BOS embedding. x has shape (b, l, d).
            # argmax returns first index of feat_token_id in case pad_token_id is
            # equal to feat_token_id.
            x = x[torch.arange(x.shape[0]),
                  (x_arg["input_ids"] == self.feat_token_id).int().argmax(dim=-1)]

        # x has shape (b, d)
        x = self.projection_head(x)
        return x


class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)
        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.audio_encoder = AudioEncoder(self.args.audio_model_id, self.args.d,
                                          self.args.freeze_encoders)
        self.image_encoder = ImageEncoder(self.args.image_model_id, self.args.d,
                                          self.args.freeze_encoders)
        self.text_encoder = TextEncoder(self.args.text_model_id, self.args.d,
                                        self.args.freeze_encoders,
                                        self.args.feat_token_id)

        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        # check if attribute exists in case model is loaded from checkpoint
        if hasattr(self.args, "freeze_logit_scale") and self.args.freeze_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init).requires_grad_(False)
        else:
            self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

    def forward(self, x):
        if self.args.use_precomputed_representations:
            r_a = self.audio_encoder(x["audio"])
            r_i = self.image_encoder(x["image"])
            r_t = self.text_encoder(x["text"])
        else:
            r_a = self.audio_encoder({"input_features": x["audio_input_features"],
                                      "attention_mask": x["audio_attention_mask"]})
            r_i = self.image_encoder({"pixel_values": x["image_pixel_values"]})
            r_t = self.text_encoder({"input_ids": x["text_input_ids"],
                                     "attention_mask": x["text_attention_mask"]})
        return r_a, r_i, r_t, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def _shared_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        if self.args.normalize:
            r_a, r_i, r_t = l2_normalize([r_a, r_i, r_t])

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp)

        return loss, logit_scale_exp

    def training_step(self, batch, batch_idx):
        loss, logit_scale_exp = self._shared_step(batch, batch_idx)

        log_n_minus_1 = np.log(len(batch[list(batch.keys())[0]]) - 1)

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n_minus_1", log_n_minus_1,
                 on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logit_scale_exp = self._shared_step(batch, batch_idx)

        log_n_minus_1 = np.log(len(batch[list(batch.keys())[0]]) - 1)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("log_n_minus_1", log_n_minus_1,
                 on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        return loss

    def on_test_start(self):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        r_i_list = []
        z_i_list = []
        idx_list = []

        for x in self.trainer.datamodule.test_dataloader():

            if self.args.use_precomputed_representations:
                r_i = self.image_encoder(x["image"].to(self.device))
            else:
                r_i = self.image_encoder(
                    {"pixel_values": x["image_pixel_values"].to(self.device)}
                    )

            r_i_list.append(r_i)
            z_i_list.append(x["z_i"].to(self.device))
            idx_list.append(x["idx"].to(self.device))

        self.z_i = torch.cat(z_i_list)
        self.r_i_idx = torch.cat(idx_list)
        r_i = torch.cat(r_i_list)

        if self.args.normalize:
            [self.r_i] = l2_normalize([r_i])
        else:
            self.r_i = r_i

    def zeroshot_accuracy(self, r_a, r_t, batch_idx):
        if self.args.loss_fn == "symile":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ MIP(r_a[i], r_i[0], r_t[i]) ... MIP(r_a[i], r_i[n-1], r_t[i]) ]
            # where MIP is the multilinear inner product.
            logits = (r_a * r_t) @ torch.t(self.r_i)
        elif self.args.loss_fn == "clip":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ r_a[i]^T r_i[0] + r_i[0]^T r_t[i]   + r_a[i]^T r_t[i] ...
            #   r_a[i]^T r_i[n-1] + r_i[n-1]^T r_t[i] + r_a[i]^T r_t[i] ]
            at = torch.diagonal(r_a @ torch.t(r_t)).unsqueeze(dim=1) # (batch_sz, 1)
            logits = at + (r_a @ torch.t(self.r_i)) + (r_t @ torch.t(self.r_i))

        logits = self.logit_scale.exp() * logits

        # pred_idx is a tensor of length batch_sz where each element is the
        # index of the r_i (across the whole test set) that maximizes the score.
        pred_idx = torch.argmax(logits, dim=1)

        # for each index in pred_idx, we get the label (0 or 1) that corresponds
        # to the r_i at that index; so pred is a tensor of length batch_sz where
        # each element is the predicted label (0 or 1)
        pred = self.z_i[pred_idx]

        # roundabout way to get true labels in case self.r_i_idx is not in order
        matching_indices = torch.nonzero(
            self.r_i_idx.unsqueeze(1) == batch_idx.unsqueeze(0), as_tuple=False)
        y = self.z_i[matching_indices[:, 0]]

        return (torch.sum(y == pred) / len(y)).item()

    def test_step(self, batch, batch_idx):
        """
        The zeroshot task is to predict which r_i corresponds to a given r_a, r_t.
        """
        r_a, _, r_t, _ = self(batch)

        if self.args.normalize:
            r_a, r_t = l2_normalize([r_a, r_t])

        zeroshot_acc = self.zeroshot_accuracy(r_a, r_t, batch["idx"])

        self.log("test_accuracy", zeroshot_acc, sync_dist=True, prog_bar=True)