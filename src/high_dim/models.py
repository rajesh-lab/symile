from argparse import Namespace
import json

import lightning.pytorch as pl
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel, WhisperModel, XLMRobertaModel

from src.losses import clip, symile
from src.utils import l2_normalize


class AudioEncoder(nn.Module):
    def __init__(self, model_id, d):
        super().__init__()
        enc = WhisperModel.from_pretrained(model_id).encoder

        self.fc = nn.Linear(enc.config.hidden_size, d, bias=True)
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
        x = x.mean(dim=1)
        x = self.layer_norm(x)
        return x


class ImageEncoder(nn.Module):
    def __init__(self, model_id, d):
        super().__init__()
        enc = CLIPVisionModel.from_pretrained(model_id)

        self.fc = nn.Linear(enc.config.hidden_size, d, bias=True)
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
    def __init__(self, model_id, d):
        super().__init__()
        if model_id == "bert-base-multilingual-cased":
            self.encoder = BertModel.from_pretrained(model_id)
        elif model_id == "xlm-roberta-base":
            self.encoder = XLMRobertaModel.from_pretrained(model_id)

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
        x = self.encoder(**x)
        x = x[1] # get pooled output
        x = self.layer_norm(x)
        return x


class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)
        self.loss_fn = symile if self.args.loss_fn == "symile" else clip

        self.audio_encoder = AudioEncoder(self.args.audio_model_id, self.args.d)
        self.image_encoder = ImageEncoder(self.args.image_model_id, self.args.d)
        self.text_encoder = TextEncoder(self.args.text_model_id, self.args.d)

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

        # r_a, r_i, r_t = l2_normalize([r_a, r_i, r_t])

        return r_a, r_i, r_t, logit_scale_exp

    def training_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self._shared_step(batch, batch_idx)

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp, self.args.efficient_loss)

        log_n_minus_1 = np.log(len(batch[list(batch.keys())[0]]) - 1)

        # zeroshot_acc = self.zeroshot_accuracy(r_a, r_t, batch["idx"], batch["lang"], batch["cls"], "train", lang_and_cls_acc=False)

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n_minus_1", log_n_minus_1,
                 on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        # self.log("train_accuracy", zeroshot_acc, sync_dist=True, prog_bar=False)
        # self.log_dict(cls_acc, on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self._shared_step(batch, batch_idx)

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp, self.args.efficient_loss)

        # log_n_minus_1 = np.log(len(batch[list(batch.keys())[0]]) - 1)

        # zeroshot_acc = self.zeroshot_accuracy(
        #     r_a, r_t, batch["idx"], batch["lang"], batch["cls"], "val", lang_and_cls_acc=False
        # )

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        # self.log("val_accuracy", zeroshot_acc,
        #          on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)

        return loss

    # def on_fit_start(self):
    #     """
    #     Compute or get r_i, which is the image representations for all data samples.
    #     """
        # self.save_image_representations()

    def save_heatmap(self, batch_cls, r_a, r_i, r_t, logit_scale_exp):
        classes = CLASSES[self.args.n_classes]

        # get indices that would sort r_a, r_i, r_t based on class id
        batch_cls_id = torch.tensor([classes[c] for c in batch_cls])
        sorted_indices = torch.argsort(batch_cls_id) # (bsz)

        # use sorted_indices to reorder r_a, r_i, r_t
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
        fig = px.imshow(logits.cpu().numpy(), aspect="auto",
                        color_continuous_scale="blues")
        fig.write_image(self.args.save_dir / "logits.png")

        ### SAVE PAIRWISE HEATMAPS ###
        ai = r_a @ torch.t(r_i) # (bsz, bsz)
        at = r_a @ torch.t(r_t) # (bsz, bsz)
        it = r_i @ torch.t(r_t) # (bsz, bsz)

        fig = px.imshow(ai.cpu().numpy(), aspect="auto", color_continuous_scale="blues")
        fig.write_image(self.args.save_dir / "dotproduct_ai.png")
        fig = px.imshow(at.cpu().numpy(), aspect="auto", color_continuous_scale="blues")
        fig.write_image(self.args.save_dir / "dotproduct_at.png")
        fig = px.imshow(it.cpu().numpy(), aspect="auto", color_continuous_scale="blues")
        fig.write_image(self.args.save_dir / "dotproduct_it.png")

    def save_image_representations(self):
        """
        Compute or get r_i, which is the image representations for all data samples.
        """
        # NOTE: if calculating accuracy for train, then make sure that dataloader shuffle is False!
        # train
        r_i_list = []
        cls_id_list = []
        idx_list = []
        for x in self.trainer.datamodule.train_dataloader():
            r_i = self.image_encoder(x["image"].to(self.device))

            r_i_list.append(r_i)
            cls_id_list.append(x["cls_id"])
            idx_list.append(x["idx"])
        self.r_i_train = torch.cat(r_i_list)
        self.cls_id_train = torch.cat(cls_id_list).to(self.device)
        self.r_i_idx_train = torch.cat(idx_list).to(self.device)

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

        # [r_i] = l2_normalize([r_i])

    def acc_per_lang(self, y_true, y_pred, batch_lang):
        acc_per_lang = {}

        for lang in ["ar", "en", "ja", "ko", "uk"]:
            lang_indices = np.array(batch_lang) == lang
            lang_pred = y_pred[lang_indices]
            lang_true = y_true[lang_indices]

            correct_predictions = torch.sum((lang_pred == lang_true).to(torch.int))
            total_samples = len(lang_pred)

            if total_samples > 0:
                acc_per_lang[lang + "_acc"] = (correct_predictions / total_samples).item()
            else:
                acc_per_lang[lang + "_acc"] = 0.0  # avoid division by zero if no samples

        return acc_per_lang

    def acc_per_class(self, y_true, y_pred, batch_cls):
        data_ref = json.load(open(self.args.data_reference))

        acc_per_cls = {}

        for cls in data_ref.keys():
            cls_indices = np.array(batch_cls) == cls
            cls_pred = y_pred[cls_indices]
            cls_true = y_true[cls_indices]

            correct_predictions = torch.sum((cls_pred == cls_true).to(torch.int))
            total_samples = len(cls_pred)

            if total_samples > 0:
                acc_per_cls[cls + "_acc"] = (correct_predictions / total_samples).item()
            else:
                acc_per_cls[cls + "_acc"] = 0.0  # avoid division by zero if no samples

        return acc_per_cls

    def zeroshot_accuracy(self, r_a, r_t, batch_idx, batch_lang, batch_cls, split, lang_and_cls_acc=False):
        # NOTE: if calculating accuracy for train, then make sure that dataloader shuffle is False!
        if split == "train":
            r_i = self.r_i_train
            cls_id = self.cls_id_train
            r_i_idx = self.r_i_idx_train
        elif split == "val":
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

        if lang_and_cls_acc:
            lang_acc = self.acc_per_lang(y, pred, batch_lang)
            cls_acc = self.acc_per_class(y, pred, batch_cls)
            return zeroshot_acc, lang_acc, cls_acc
        else:
            return zeroshot_acc

    def test_step(self, batch, batch_idx):
        """
        The zeroshot task is to predict which r_i corresponds to a given r_a, r_t.
        """
        r_a, r_i, r_t, logit_scale_exp = self._shared_step(batch, batch_idx)

        # save heatmap of the logits for first batch
        if batch_idx == 0 and self.args.save_test_heatmap:
            self.save_heatmap(batch["cls"], r_a, r_i, r_t, logit_scale_exp)

        # zeroshot_acc, lang_acc, cls_acc = self.zeroshot_accuracy(
        #     r_a, r_t, batch["idx"], batch["lang"], batch["cls"], "test", lang_and_cls_acc=True
        # )

        # self.log("test_accuracy", zeroshot_acc, sync_dist=True, prog_bar=True)
        # self.log_dict(lang_acc, on_step=True, on_epoch=True)
        # self.log_dict(cls_acc, on_step=True, on_epoch=True)