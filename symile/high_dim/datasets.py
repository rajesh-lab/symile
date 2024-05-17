import os

import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLMRobertaTokenizer, MT5Tokenizer

from symile.high_dim.constants import MISSING_TOKEN
from symile.high_dim.utils import get_language_constant


class HighDimDataset(Dataset):
    def __init__(self, args, split, txt_tokenizer=None):
        self.args = args
        self.split = split
        self.txt_tokenizer = txt_tokenizer

        self.split_dir = self.args.data_dir / f"{split}"

        self.text_input_ids = torch.load(self.split_dir / f"text_input_ids_{split}.pt").long()
        self.text_attention_mask = torch.load(self.split_dir / f"text_attention_mask_{split}.pt")
        if self.args.text_model_id == "bert-base-multilingual-cased":
            self.text_token_type_ids = torch.load(self.split_dir / f"text_token_type_ids_{split}.pt").long()
        self.max_token_len = self.text_input_ids.shape[1]

        self.image = torch.load(self.split_dir / f"image_{split}.pt")
        self.image_mean = torch.mean(self.image, dim=0)

        self.audio = torch.load(self.split_dir / f"audio_{split}.pt")
        self.audio_mean = torch.mean(self.audio, dim=0)

        self.cls_id = torch.load(self.split_dir / f"cls_id_{split}.pt")
        self.idx = torch.load(self.split_dir / f"idx_{split}.pt")

        with open(self.split_dir / f"lang_{split}.txt", "r") as f:
            self.lang = f.read().splitlines()

        self.languages = get_language_constant(self.args.num_langs)

        if getattr(self.args, "missingness", False) and self.split != "test":
            if self.args.missingness_prob == 0.5:
                missingness_prob_str = "50"
            elif self.args.missingness_prob == 0.75:
                missingness_prob_str = "75"
            else:
                raise ValueError("Missingness probability not supported.")
            self.text_missingness = torch.load(self.split_dir / f"text_missingness_prob{missingness_prob_str}_{split}.pt")
            self.image_missingness = torch.load(self.split_dir / f"image_missingness_prob{missingness_prob_str}_{split}.pt")
            self.audio_missingness = torch.load(self.split_dir / f"audio_missingness_prob{missingness_prob_str}_{split}.pt")

    def __len__(self):
        return len(self.image)

    def get_missingness_text(self):
        encoded_inputs = self.txt_tokenizer(text=MISSING_TOKEN,
                                            return_tensors="pt",
                                            padding="max_length",
                                            max_length=self.max_token_len)
        encoded_inputs["input_ids"] = torch.squeeze(encoded_inputs["input_ids"], dim=0)
        encoded_inputs["attention_mask"] = torch.squeeze(encoded_inputs["attention_mask"], dim=0)
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].to(torch.float32)
        return encoded_inputs

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
        if self.args.text_model_id == "bert-base-multilingual-cased":
            text = {"input_ids": self.text_input_ids[idx],
                    "token_type_ids": self.text_token_type_ids[idx],
                    "attention_mask": self.text_attention_mask[idx]}
        else:
            text = {"input_ids": self.text_input_ids[idx],
                    "attention_mask": self.text_attention_mask[idx]}

        image = self.image[idx]
        audio = self.audio[idx]

        text_missing = 0
        image_missing = 0
        audio_missing = 0

        if getattr(self.args, "missingness", False) and self.split != "test":
            text_missing = self.text_missingness[idx]
            image_missing = self.image_missingness[idx]
            audio_missing = self.audio_missingness[idx]

            if text_missing == 1:
                text = self.get_missingness_text()

            if image_missing == 1:
                image = self.image_mean

            if audio_missing == 1:
                audio = self.audio_mean

        if (text_missing == 0) and (image_missing == 0) and (audio_missing == 0):
            all_observed = 1
        else:
            all_observed = 0

        return {"text": text,
                "image": image,
                "audio": audio,
                "cls_id": self.cls_id[idx],
                "idx": self.idx[idx],
                "lang_id": self.languages.index(self.lang[idx]),
                "text_missing": text_missing,
                "image_missing": image_missing,
                "audio_missing": audio_missing,
                "all_observed": all_observed}


class HighDimDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

        self.txt_tokenizer = None
        if self.args.missingness:
            self.txt_tokenizer = self.get_tokenizer()
            if MISSING_TOKEN not in self.txt_tokenizer.get_vocab():
                self.txt_tokenizer.add_tokens([MISSING_TOKEN])
                self.tokenizer_len = len(self.txt_tokenizer)

    def get_tokenizer(self):
        if self.args.text_model_id == "bert-base-multilingual-cased":
            return BertTokenizer.from_pretrained(self.args.text_model_id)
        elif self.args.text_model_id == "xlm-roberta-base" or self.args.text_model_id == "xlm-roberta-large":
            return XLMRobertaTokenizer.from_pretrained(self.args.text_model_id)
        elif self.args.text_model_id == "google/mt5-base" or self.args.text_model_id == "google/mt5-small" or self.args.text_model_id == "google/mt5-large" or self.args.text_model_id == "google/mt5-xxl":
            return MT5Tokenizer.from_pretrained(self.args.text_model_id)

    def setup(self, stage):
        self.ds_train = HighDimDataset(self.args, "train", self.txt_tokenizer)
        self.ds_val = HighDimDataset(self.args, "val", self.txt_tokenizer)
        self.ds_test = HighDimDataset(self.args, "test", self.txt_tokenizer)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=False)