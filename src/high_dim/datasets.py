import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, CLIPImageProcessor, \
                         WhisperFeatureExtractor, XLMRobertaTokenizer, \
                         MT5EncoderModel, T5Tokenizer


from src.high_dim.constants import LANGUAGES


class HighDimDataset(Dataset):
    def __init__(self, df, data_dir, split):
        self.df = df
        self.split_dir = data_dir / f"{split}"

    def __len__(self):
        return len(self.df)

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
                - lang: (str) with language ISO-639 code for audio sample.
                - cls: (str) with class label for image sample.
                - target_text: (str) with target text.
                - idx: (int) unique identifier for data sample.
        """
        audio = np.load(self.split_dir / f"audio/{self.df.iloc[idx].audio_filename}.npy")

        image = np.load(self.split_dir / f"image/{self.df.iloc[idx].image_filename}.npy")

        text = self.df.iloc[idx].text

        cls = self.df.iloc[idx].cls
        cls_id = self.df.iloc[idx].cls_id

        lang = self.df.iloc[idx].lang
        lang_id = LANGUAGES[lang]

        target_text = self.df.iloc[idx].target_text

        return {"audio": audio, "image": image, "text": text,
                "cls": cls, "cls_id": cls_id,
                "lang": lang, "lang_id": lang_id,
                "target_text": target_text,
                "idx": idx}

class Collator:
    """
    Custom collate function so that the text tokenizer can be called on a batch
    of text, which is then padded to the length of the longest sequence in the batch.
    """
    def __init__(self, txt_tokenizer):
        self.txt_tokenizer = txt_tokenizer
    def __call__(self, batch):
        """
        Args:
            batch (list): List of data samples of length `batch_sz`. Each sample
                          is a dictionary with keys `audio`, `image`, `text`,
                          `z_a`, `z_i`, `z_t`, and `idx`
                          (see SymileDataset.__getitem__).
        Returns:
            (dict): of batched data samples with the following keys:
                - audio_input_features: torch.Tensor of shape (batch_sz, 80, 3000)
                - audio_attention_mask: torch.Tensor of shape (batch_sz, 3000)
                - image_pixel_values: torch.Tensor of shape (batch_sz, 3, 224, 224)
                - text_input_ids: torch.Tensor of shape (batch_sz, len_longest_seq)
                - text_attention_mask: torch.Tensor of shape (batch_sz, len_longest_seq)
                - z_a: torch.Tensor of shape (batch_sz) with z_a used in data generating process
                - z_i: torch.Tensor of shape (batch_sz) with z_i used in data generating process
                - z_t: torch.Tensor of shape (batch_sz) with z_t used in data generating process
                - idx: torch.Tensor of shape (batch_sz) with unique identifier for data sample
        """
        audio = torch.tensor(np.stack([s["audio"] for s in batch]))

        image = torch.tensor(np.stack([s["image"] for s in batch]))

        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding=True, truncation=True)

        lang = [s["lang"] for s in batch]
        lang_id = torch.tensor([s["lang_id"] for s in batch])

        cls = [s["cls"] for s in batch]
        cls_id = torch.tensor([s["cls_id"] for s in batch])

        target_text = [s["target_text"] for s in batch]

        idx = torch.tensor([s["idx"] for s in batch])

        return {"audio": audio,
                "image": image,
                "text": text,
                "lang": lang,
                "lang_id": lang_id,
                "cls": cls,
                "cls_id": cls_id,
                "target_text": target_text,
                "idx": idx}


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def text_tokenization(self):
        """
        Gets text tokenizer and sets feat_token_id attribute.
        Note that encoder features are taken from the EOS or BOS embedding.

        transformers' CLIP and ImageBind take features from EOS embedding:
        - https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/models/clip/modeling_clip.py#L757C4-L757C4
        - https://github.com/facebookresearch/ImageBind/blob/95d27c7fd5a8362f3527e176c3a80ae5a4d880c0/imagebind/models/imagebind_model.py#L384C9-L384C9

        mBERT and XLM-Roberta take features from BOS embedding:
        - https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/models/bert/modeling_bert.py#L661
        - https://github.com/huggingface/transformers/blob/41aef33758ae166291d72bc381477f2db84159cf/src/transformers/models/xlm_roberta/modeling_xlm_roberta.py#L580
        """
        if self.args.text_model_id == "bert-base-multilingual-cased":
            self.txt_tokenizer = BertTokenizer.from_pretrained(self.args.text_model_id)
        elif self.args.text_model_id == "xlm-roberta-base":
            self.txt_tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.text_model_id)
        elif self.args.text_model_id == "google/mt5-base" or self.args.text_model_id == "google/mt5-small":
            self.txt_tokenizer =tokenizer = T5Tokenizer.from_pretrained(self.args.text_model_id)


class HighDimDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.text_tokenization()

        df_train = pd.read_csv(self.args.data_dir / self.args.train_csv)
        self.ds_train = HighDimDataset(df_train, self.args.data_dir, "train")

        df_val = pd.read_csv(self.args.data_dir / self.args.val_csv)
        self.ds_val = HighDimDataset(df_val, self.args.data_dir, "val")

        df_test = pd.read_csv(self.args.data_dir / self.args.test_csv)
        self.ds_test = HighDimDataset(df_test, self.args.data_dir, "test")

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=self.args.drop_last)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer),
                          drop_last=self.args.drop_last)