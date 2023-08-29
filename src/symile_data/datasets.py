from PIL import Image
import torch
import torchaudio
from torch.utils.data import Dataset


class SymileDataset(Dataset):
    def __init__(self, df, audio_feat_extractor, img_processor):
        self.df = df
        self.audio_feat_extractor = audio_feat_extractor
        self.img_processor = img_processor

    def __len__(self):
        return len(self.df)

    def get_audio(self, path):
        # downsample to 16kHz, as expected by Whisper, before passing to feature extractor
        waveform, sr = torchaudio.load(path)
        resampler = torchaudio.transforms.Resample(sr, self.audio_feat_extractor.sampling_rate)
        waveform = torch.squeeze(resampler(waveform))
        audio = self.audio_feat_extractor(
                        waveform,
                        return_attention_mask=True,
                        return_tensors="pt",
                        sampling_rate=self.audio_feat_extractor.sampling_rate,
                        do_normalize=True
                    )
        return {"input_features": torch.squeeze(audio.input_features),
                "attention_mask": torch.squeeze(audio.attention_mask)}

    def get_image(self, path):
        image = Image.open(path)
        image = self.img_processor(images=image, return_tensors="pt")
        return {"pixel_values": torch.squeeze(image.pixel_values)}

    def __getitem__(self, idx):
        """
        Returns:
            (dict): containing the following key-value pairs:
                - audio: (dict) whose key-value pairs are
                    (input_features: torch.Tensor of shape (80, 3000)) and
                    (attention_mask: torch.Tensor of shape (3000)).
                - image: (dict) whose key-value pairs are
                    (pixel_values: torch.Tensor of shape (3, 224, 224)).
                - text: (str) with data sample text.
                - template: (int) with data sample template number.
        """
        audio = self.get_audio(self.df.iloc[idx].audio_path)
        image = self.get_image(self.df.iloc[idx].image_path)
        text = self.df.iloc[idx].text
        template = self.df.iloc[idx].template
        return {"audio": audio, "image": image, "text": text, "template": template}