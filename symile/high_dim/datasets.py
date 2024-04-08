import os

import lightning.pytorch as pl
import torch
from torch.utils.data import Dataset, DataLoader


from src.high_dim.constants import LANGUAGES


class HighDimDataset(Dataset):
    def __init__(self, args, split):
        self.args = args

        self.split_dir = self.args.data_dir / f"{split}"

        self.text_input_ids = torch.load(self.split_dir / f"text_input_ids_{split}.pt").long()
        self.text_attention_mask = torch.load(self.split_dir / f"text_attention_mask_{split}.pt")
        if self.args.text_model_id == "bert-base-multilingual-cased":
            self.text_token_type_ids = torch.load(self.split_dir / f"text_token_type_ids_{split}.pt").long()

        self.image = torch.load(self.split_dir / f"image_{split}.pt")
        self.audio = torch.load(self.split_dir / f"audio_{split}.pt")

        self.cls_id = torch.load(self.split_dir / f"cls_id_{split}.pt")
        self.idx = torch.load(self.split_dir / f"idx_{split}.pt")

        with open(self.split_dir / f"lang_{split}.txt", "r") as f:
            self.lang = f.read().splitlines()
        with open(self.split_dir / f"cls_{split}.txt", "r") as f:
            self.cls = f.read().splitlines()

    def __len__(self):
        return len(self.image)

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

        return {"text": text,
                "image": self.image[idx],
                "audio": self.audio[idx],
                "cls_id": self.cls_id[idx],
                "idx": self.idx[idx],
                "lang": self.lang[idx],
                "lang_id": LANGUAGES[self.lang[idx]],
                "cls": self.cls[idx]}


class HighDimDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def setup(self, stage):
        self.ds_train = HighDimDataset(self.args, "train")
        self.ds_val = HighDimDataset(self.args, "val")
        self.ds_test = HighDimDataset(self.args, "test")

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz_train,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          shuffle=False,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)