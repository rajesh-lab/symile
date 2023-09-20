import os

import lightning.pytorch as pl
import pandas as pd
from PIL import Image
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer, CLIPImageProcessor, \
                         WhisperFeatureExtractor, XLMRobertaTokenizer


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
                - idx: (int) unique identifier for data sample.
                - support (optional): (int) 1 if data sample is in support, 0 otherwise.
        """
        audio = self.get_audio(self.df.iloc[idx].audio_path)
        image = self.get_image(self.df.iloc[idx].image_path)
        text = self.df.iloc[idx].text
        template = self.df.iloc[idx].template

        item_dict = {"audio": audio, "image": image, "text": text,
                     "template": template, "idx": idx}

        if "in_support" in self.df:
            item_dict["in_support"] = self.df.iloc[idx].in_support

        return item_dict


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
                          `template`, `idx`, and (optionally) `in_support`
                          (see SymileDataset.__getitem__).
        Returns:
            (dict): of batched data samples with the following keys:
                - audio_input_features: torch.Tensor of shape (batch_sz, 80, 3000)
                - audio_attention_mask: torch.Tensor of shape (batch_sz, 3000)
                - image_pixel_values: torch.Tensor of shape (batch_sz, 3, 224, 224)
                - text_input_ids: torch.Tensor of shape (batch_sz, len_longest_seq)
                - text_attention_mask: torch.Tensor of shape (batch_sz, len_longest_seq)
                - templates: torch.Tensor of shape (batch_sz) containing template numbers
                - idx: torch.Tensor of shape (batch_sz) with unique identifier for data sample
                - in_support: (optional) torch.Tensor of shape (batch_sz) where 1 means
                              sample is in support, 0 otherwise.
        """
        audio_input_features = torch.stack([s["audio"]["input_features"] for s in batch])
        audio_attention_mask = torch.stack([s["audio"]["attention_mask"] for s in batch])

        image_pixel_values = torch.stack([s["image"]["pixel_values"] for s in batch])

        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding=True, truncation=True)

        templates = torch.Tensor([s["template"] for s in batch])
        idx = torch.Tensor([s["idx"] for s in batch])

        batched_data = {"audio_input_features": audio_input_features,
                        "audio_attention_mask": audio_attention_mask,
                        "image_pixel_values": image_pixel_values,
                        "text_input_ids": text["input_ids"],
                        "text_attention_mask": text["attention_mask"],
                        "templates": templates,
                        "idx": idx}

        if "in_support" in batch[0]:
            batched_data["in_support"] = torch.Tensor([s["in_support"] for s in batch])

        return batched_data


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.audio_feat_extractor = WhisperFeatureExtractor.from_pretrained(args.audio_model_id)
        self.img_processor = CLIPImageProcessor.from_pretrained(args.image_model_id)

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def get_full_data_paths(self, df):
        """
        Update audio_path and image_path in dataframe to use appropriate
        parent directories.
        """
        def _full_audio_path(r):
            if r.template in [1]:
                return self.args.data_dir_commonvoice / r.audio_path.strip("/")
            else:
                return self.args.data_dir_generated_audio / r.audio_path.strip("/")
        df["audio_path"] = df.apply(lambda r: _full_audio_path(r), axis=1)

        def _full_image_path(r):
            if r.template in [1, 3]:
                return self.args.data_dir_imagenet / r.image_path.strip("/")
            else:
                return self.args.data_dir_flags / r.image_path.strip("/")
        df["image_path"] = df.apply(lambda r: _full_image_path(r), axis=1)

        return df

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
            if self.args.text_embedding == "eos":
                self.feat_token_id = self.txt_tokenizer.sep_token_id
            elif self.args.text_embedding == "bos":
                self.feat_token_id = self.txt_tokenizer.cls_token_id
        elif self.args.text_model_id == "xlm-roberta-base":
            self.txt_tokenizer = XLMRobertaTokenizer.from_pretrained(self.args.text_model_id)
            if self.args.text_embedding == "eos":
                self.feat_token_id = self.txt_tokenizer.eos_token_id
            elif self.args.text_embedding == "bos":
                self.feat_token_id = self.txt_tokenizer.bos_token_id


class PretrainDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.text_tokenization()

        df_train = pd.read_csv(self.args.train_dataset_path)
        df_train = self.get_full_data_paths(df_train)
        self.ds_train = SymileDataset(df_train, self.audio_feat_extractor, self.img_processor)

        df_val = pd.read_csv(self.args.val_dataset_path)
        df_val = self.get_full_data_paths(df_val)
        self.ds_val = SymileDataset(df_val, self.audio_feat_extractor, self.img_processor)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer))

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer))


class SupportClfDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.text_tokenization()

        df_train = pd.read_csv(self.args.support_train_dataset_path)
        df_train = self.get_full_data_paths(df_train)
        self.ds_train = SymileDataset(df_train, self.audio_feat_extractor, self.img_processor)

        df_val = pd.read_csv(self.args.support_test_dataset_path)
        df_val = self.get_full_data_paths(df_val)
        self.ds_val = SymileDataset(df_val, self.audio_feat_extractor, self.img_processor)

        df_test = pd.read_csv(self.args.support_test_dataset_path)
        df_test = self.get_full_data_paths(df_test)
        self.ds_test = SymileDataset(df_test, self.audio_feat_extractor, self.img_processor)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer))

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer))

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer))


class ZeroshotClfDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.text_tokenization()

        df_test = pd.read_csv(self.args.zeroshot_dataset_path)
        df_test = self.get_full_data_paths(df_test)
        self.ds_test = SymileDataset(df_test, self.audio_feat_extractor, self.img_processor)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          num_workers=self.num_workers,
                          collate_fn=Collator(self.txt_tokenizer))