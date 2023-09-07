from argparse import Namespace

import lightning.pytorch as pl
import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel, WhisperModel, XLMRobertaModel
try:
    import wandb
except ImportError:
    wandb = None

from src.losses import pairwise_infonce, symile
from utils import l2_normalize


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
    def __init__(self, model_id, d):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(model_id).encoder
        self.projection_head = ProjectionHead(self.encoder.config.hidden_size, d,
                                              self.encoder.layer_norm.eps)

    def forward(self, input_features, attention_mask):
        """
        Args:
            input_features (torch.Tensor): shape (batch_sz, 80, 3000)
            attention_mask (torch.Tensor): shape (batch_sz, 3000)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.encoder(input_features=input_features, attention_mask=attention_mask)
        x = x["last_hidden_state"]

        # select first embedding as done by ImageBind:
        # https://github.com/facebookresearch/ImageBind/blob/95d27c7fd5a8362f3527e176c3a80ae5a4d880c0/imagebind/models/imagebind_model.py#L391C3-L391C3
        x = x[:, 0, :]

        x = self.projection_head(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, model_id, d):
        super().__init__()
        self.encoder = CLIPVisionModel.from_pretrained(model_id)
        self.projection_head = ProjectionHead(self.encoder.config.hidden_size, d,
                                              self.encoder.config.layer_norm_eps)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values (torch.Tensor): shape (batch_sz, 3, 224, 224)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.encoder(pixel_values=pixel_values)
        x = x["last_hidden_state"]

        # select first embedding as done by transformers' CLIP and ImageBind:
        # https://github.com/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/clip/modeling_clip.py#L894
        # https://github.com/facebookresearch/ImageBind/blob/95d27c7fd5a8362f3527e176c3a80ae5a4d880c0/imagebind/models/imagebind_model.py#L380
        x = x[:, 0, :]

        x = self.projection_head(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self, model_id, d, feat_token_id):
        super().__init__()
        self.feat_token_id = feat_token_id
        if model_id == "bert-base-multilingual-cased":
            self.encoder = BertModel.from_pretrained(model_id)
        elif model_id == "xlm-roberta-base":
            self.encoder = XLMRobertaModel.from_pretrained(model_id)
        self.projection_head = ProjectionHead(self.encoder.config.hidden_size, d,
                                              self.encoder.config.layer_norm_eps)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids (torch.Tensor): shape (batch_sz, len_longest_seq)
            attention_mask (torch.Tensor): shape (batch_sz, len_longest_seq)
        Returns:
            x (torch.Tensor): shape (batch_sz, d)
        """
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        x = x["last_hidden_state"]

		# take features from EOS or BOS embedding. x has shape (b, l, d).
        # argmax returns first index of feat_token_id in case pad_token_id is
        # equal to feat_token_id.
        x = x[torch.arange(x.shape[0]),
              (input_ids == self.feat_token_id).int().argmax(dim=-1)]

        # x has shape (b, d)
        x = self.projection_head(x)
        return x


class SymileModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.args = Namespace(**args)
        self.loss_fn = symile if self.args.loss_fn == "symile" else pairwise_infonce

        self.audio_encoder = AudioEncoder(self.args.audio_model_id, self.args.d)
        self.image_encoder = ImageEncoder(self.args.image_model_id, self.args.d)
        self.text_encoder = TextEncoder(self.args.text_model_id, self.args.d,
                                        self.args.feat_token_id)
        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        self.logit_scale = nn.Parameter(torch.ones([]) * self.args.logit_scale_init)

        for k in args.keys():
            self.save_hyperparameters(k)

    def forward(self, x):
        r_a = self.audio_encoder(x["audio_input_features"], x["audio_attention_mask"])
        r_i = self.image_encoder(x["image_pixel_values"])
        r_t = self.text_encoder(x["text_input_ids"], x["text_attention_mask"])
        return r_a, r_i, r_t, self.logit_scale.exp()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def _shared_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        if self.args.normalize:
            r_a, r_i, r_t = l2_normalize([r_a, r_i, r_t])

        return self.loss_fn(r_a, r_i, r_t, logit_scale_exp)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                      on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        return loss


class SupportClfModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.args = Namespace(**args)
        for k in args.keys():
            self.save_hyperparameters(k)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SymileModel.load_from_checkpoint(self.args.ckpt_path,
                                                      map_location=device)
        self.model.freeze()
        self.audio_encoder = self.model.audio_encoder
        self.image_encoder = self.model.image_encoder
        self.text_encoder = self.model.text_encoder

        in_features = self.text_encoder.projection_head.linear_projection.out_features
        self.classifier = nn.Linear(in_features, 1, bias=True)

        self.training_step_outputs = {"y": [], "prob": []}

    def forward(self, x):
        r_a = self.audio_encoder(x["audio_input_features"], x["audio_attention_mask"])
        r_i = self.image_encoder(x["image_pixel_values"])
        r_t = self.text_encoder(x["text_input_ids"], x["text_attention_mask"])

        if self.model.args.normalize:
            r_a, r_i, r_t = l2_normalize([r_a, r_i, r_t])

        if self.model.args.loss_fn == "symile":
            X = r_a * r_i * r_t
        elif self.model.args.loss_fn == "pairwise_infonce":
            if self.args.concat_infonce:
                X = torch.cat((r_a * r_i, r_i * r_t, r_a * r_t), dim=1)
            else:
                X = (r_a * r_i) + (r_i * r_t) + (r_a * r_t)

        if self.args.use_logit_scale:
            X = self.model.logit_scale.exp() * X

        return self.classifier(X)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def _shared_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch["in_support"].unsqueeze(1)
        return nn.BCEWithLogitsLoss()(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        y = batch["in_support"].unsqueeze(1)
        loss = nn.BCEWithLogitsLoss()(y_hat, y)

        # save y and probabilities to compute accuracy at epoch end
        prob = nn.Sigmoid()(y_hat)
        self.training_step_outputs["prob"].append(prob)
        self.training_step_outputs["y"].append(y.squeeze())

        self.log("test_loss", loss, on_step=False, on_epoch=True,
                 sync_dist=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        prob = torch.cat(self.training_step_outputs["prob"])
        pred = torch.where(prob >= 0.5, 1, 0).squeeze()
        y = torch.cat(self.training_step_outputs["y"])
        acc = (torch.sum(y == pred) / len(y)).item()
        self.log("test_accuracy", acc, sync_dist=True, prog_bar=True)

        # wandb.plot.roc_curve expects probabilities to have shape (n, n_classes)
        if self.args.wandb:
            y = y.cpu()
            prob = prob.cpu()
            probs = torch.cat([(1-prob), prob], dim=1)
            self.logger.experiment.log({"roc_curve": wandb.plot.roc_curve(y, probs)})

        self.training_step_outputs.clear() # free memory