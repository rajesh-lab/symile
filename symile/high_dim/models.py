from argparse import Namespace
import json

import lightning.pytorch as pl
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
from transformers import BertModel, XLMRobertaModel

from src.losses import clip, symile


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

    def _shared_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        return r_a, r_i, r_t, logit_scale_exp

    def training_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self._shared_step(batch, batch_idx)

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp, self.args.efficient_loss)

        log_n = np.log(len(batch['image']))

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n", log_n,
                 on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self._shared_step(batch, batch_idx)

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp, self.args.efficient_loss)

        zeroshot_acc = self.zeroshot_accuracy(r_a, r_t, batch["idx"], batch["lang"], batch["cls"], "val")

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_accuracy", zeroshot_acc,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        return loss

    def on_fit_start(self):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        self.save_image_representations()

    def save_image_representations(self):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        # val
        r_i_list = []
        cls_id_list = []
        idx_list = []
        for x in self.trainer.datamodule.val_dataloader():
            r_i = self.image_encoder(x["image"].to(self.device))

            r_i_list.append(r_i)
            cls_id_list.append(x["cls_id"])
            idx_list.append(x["idx"])
        self.r_i_val = torch.cat(r_i_list)
        self.cls_id_val = torch.cat(cls_id_list).to(self.device)
        self.r_i_idx_val = torch.cat(idx_list).to(self.device)

        # test
        r_i_list = []
        cls_id_list = []
        idx_list = []
        for x in self.trainer.datamodule.test_dataloader():
            r_i = self.image_encoder(x["image"].to(self.device))

            r_i_list.append(r_i)
            cls_id_list.append(x["cls_id"])
            idx_list.append(x["idx"])
        self.r_i_test = torch.cat(r_i_list)
        self.cls_id_test = torch.cat(cls_id_list).to(self.device)
        self.r_i_idx_test = torch.cat(idx_list).to(self.device)

    def zeroshot_accuracy(self, r_a, r_t, batch_idx, batch_lang, batch_cls, split):
        if split == "val":
            r_i = self.r_i_val
            cls_id = self.cls_id_val
            r_i_idx = self.r_i_idx_val
        elif split == "test":
            r_i = self.r_i_test
            cls_id = self.cls_id_test
            r_i_idx = self.r_i_idx_test

        if self.args.loss_fn == "symile":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ MIP(r_a[i], r_i[0], r_t[i]) ... MIP(r_a[i], r_i[n-1], r_t[i]) ]
            # where MIP is the multilinear inner product.
            logits = (r_a * r_t) @ torch.t(r_i)
        elif self.args.loss_fn == "clip":
            # logits is a (batch_sz, n) matrix where each row i is
            # [ r_a[i]^T r_i[0] + r_i[0]^T r_t[i]   + r_a[i]^T r_t[i] ...
            #   r_a[i]^T r_i[n-1] + r_i[n-1]^T r_t[i] + r_a[i]^T r_t[i] ]
            at = torch.diagonal(r_a @ torch.t(r_t)).unsqueeze(dim=1) # (batch_sz, 1)
            logits = at + (r_a @ torch.t(r_i)) + (r_t @ torch.t(r_i))

        logits = self.logit_scale.exp() * logits

        # pred_idx is a tensor of length batch_sz where each element is the
        # index of the r_i (across the whole test set) that maximizes the score.
        pred_idx = torch.argmax(logits, dim=1)

        # for each index in pred_idx, we get the class id (label) that corresponds
        # to the r_i at that index; so pred is a tensor of length batch_sz where
        # each element is the predicted label.
        pred = cls_id[pred_idx]

        # roundabout way to get true labels in case r_i_idx is not in order
        matching_indices = torch.nonzero(
            r_i_idx.unsqueeze(1) == batch_idx.unsqueeze(0), as_tuple=False)
        y = cls_id[matching_indices[:, 0]]

        zeroshot_acc = (torch.sum(y == pred) / len(y)).item()

        return zeroshot_acc

    def save_heatmaps(self, batch, r_a, r_i, r_t, logit_scale_exp, save_dir):
        # get indices that would sort r_a, r_i, r_t based on class id
        # and reorder r_a, r_i, r_t accordingly
        sorted_indices = torch.argsort(batch["cls_id"])
        r_a = r_a[sorted_indices]
        r_i = r_i[sorted_indices]
        r_t = r_t[sorted_indices]

        ### GET LOGITS ###
        if self.args.loss_fn == "symile":
            # logits is a (bsz, bsz) matrix where each row i is
            # [ MIP(r_a[i], r_i[0], r_t[i]) ... MIP(r_a[i], r_i[bsz-1], r_t[i]) ]
            # where MIP is the multilinear inner product.
            logits = (r_a * r_t) @ torch.t(r_i)
        elif self.args.loss_fn == "clip":
            # logits is a (bsz, bsz) matrix where each row i is
            # [ r_a[i]^T r_i[0]     + r_i[0]^T r_t[i]     + r_a[i]^T r_t[i] ...
            #   r_a[i]^T r_i[bsz-1] + r_i[bsz-1]^T r_t[i] + r_a[i]^T r_t[i] ]
            at = torch.diagonal(r_a @ torch.t(r_t)).unsqueeze(dim=1) # (bsz, 1)
            logits = at + (r_a @ torch.t(r_i)) + (r_t @ torch.t(r_i))

        logits = logit_scale_exp * logits

        ### SAVE HEATMAP ###
        fig = px.imshow(logits.cpu().detach().numpy(), aspect="auto",
                        color_continuous_scale="blues")
        fig.write_image(save_dir / "logits.png")

        ### SAVE PAIRWISE HEATMAPS ###
        ai = r_a @ torch.t(r_i) # (bsz, bsz)
        at = r_a @ torch.t(r_t) # (bsz, bsz)
        it = r_i @ torch.t(r_t) # (bsz, bsz)

        fig = px.imshow(ai.cpu().detach().numpy(), aspect="auto", color_continuous_scale="blues")
        fig.write_image(save_dir / "dotproduct_ai.png")
        fig = px.imshow(at.cpu().detach().numpy(), aspect="auto", color_continuous_scale="blues")
        fig.write_image(save_dir / "dotproduct_at.png")
        fig = px.imshow(it.cpu().detach().numpy(), aspect="auto", color_continuous_scale="blues")
        fig.write_image(save_dir / "dotproduct_it.png")

    def test_step(self, batch, batch_idx, save_test_heatmaps=False, save_dir=None):
        """
        The zeroshot task is to predict which r_i corresponds to a given r_a, r_t.
        """
        r_a, r_i, r_t, logit_scale_exp = self._shared_step(batch, batch_idx)

        if save_test_heatmaps:
            # save heatmap of the logits for first batch
            self.save_heatmaps(batch, r_a, r_i, r_t, logit_scale_exp, save_dir)
        else:
            zeroshot_acc = self.zeroshot_accuracy(r_a, r_t, batch["idx"], batch["lang"],
                                                batch["cls"], "test")
            self.log("test_accuracy", zeroshot_acc, sync_dist=True, prog_bar=True)