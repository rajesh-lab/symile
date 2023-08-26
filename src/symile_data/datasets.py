import librosa
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, WhisperFeatureExtractor, XLMRobertaTokenizer


class SymileDataset(Dataset):
    """
    TODO: add comments
    """
    def __init__(self, args, df):
        self.df = df

        # AUDIO VARIABLES
        self.audio_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)
        self.whisper_sampling_rate = 16000

        # TEXT VARIABLES
        self.txt_tokenizer = XLMRobertaTokenizer.from_pretrained(args.text_model_id)

        # IMAGE VARIABLES
        self.img_processor = CLIPImageProcessor.from_pretrained(args.image_model_id)


    def __len__(self):
        """
        Compute length of the dataset.

        Returns:
            (int): dataset size.
        """
        return len(self.df)

    def get_audio(self, path):
        """
        TODO: for when you write comments, https://huggingface.co/blog/fine-tune-whisper
        """
        # downsample to 16kHz, as expected by Whisper, before passing to feature extractor
        waveform, _ = librosa.load(path, sr=self.whisper_sampling_rate)
        return self.audio_feat_extractor(
                                waveform,
                                return_attention_mask=True,
                                return_tensors="pt",
                                sampling_rate=self.whisper_sampling_rate,
                                do_normalize=True
                            )

    def get_image(self, path):
        image = Image.open(path)
        return self.img_processor(images=image, return_tensors="pt")

    def get_text(self, text):
        return self.txt_tokenizer(text=text, return_tensors="pt")

    def __getitem__(self, idx):
        """
        Index into the dataset.

        Args:
            idx (int): index of data sample to retrieve.
        Returns: TODO
        """
        template = self.df.iloc[idx].template
        text = self.get_text(self.df.iloc[idx].text)
        audio = self.get_audio(self.df.iloc[idx].audio_path)
        image = self.get_image(self.df.iloc[idx].image_path)
        breakpoint()
        return {"text": text, "audio": audio, "image": image, "template": template}