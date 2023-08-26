import torch
import torch.nn as nn
from transformers import CLIPVisionModel, WhisperModel, XLMRobertaModel


class AudioEncoder(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model = WhisperModel.from_pretrained(model_id).encoder

    def forward(self, x):
        return self.model(x)


class ImageEncoder(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_id)

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.model = XLMRobertaModel.from_pretrained(model_id)

    def forward(self, x):
        return self.model(x)


class SymileModel(nn.Module):
    def __init__(self, audio_encoder, image_encoder, text_encoder, logit_scale_init):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # temperature parameter is learned as done by CLIP:
        # https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/model.py#L295
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

    def forward(self, audio, image, text):
        r_a = self.audio_encoder(audio)
        r_i = self.image_encoder(image)
        r_t = self.text_encoder(text)
        return r_a, r_i, r_t, self.logit_scale.exp()