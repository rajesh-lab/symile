import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel, WhisperModel, XLMRobertaModel


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


class SymileModel(nn.Module):
    def __init__(self, audio_encoder, image_encoder, text_encoder, logit_scale_init):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

    def forward(self, x):
        r_a = self.audio_encoder(x["audio_input_features"], x["audio_attention_mask"])
        r_i = self.image_encoder(x["image_pixel_values"])
        r_t = self.text_encoder(x["text_input_ids"], x["text_attention_mask"])
        return r_a, r_i, r_t, self.logit_scale.exp()