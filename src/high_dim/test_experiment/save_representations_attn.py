import os

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLMRobertaTokenizer, \
                         BertModel, XLMRobertaModel, \
                         WhisperFeatureExtractor, WhisperModel, \
                         MT5EncoderModel, T5Tokenizer
from tqdm import tqdm

from args_attn import parse_args_save_representations


class HighDimDataset(Dataset):
    def __init__(self, df):
        self.df = df

        langs = sorted(df["lang"].unique())
        assert len(langs) == 5, "Expected 5 languages in dataset."

        self.lang_embeddings = {value: idx for idx, value in enumerate(langs)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        lang_embed = self.lang_embeddings[self.df.iloc[idx].lang]
        cls_id = self.df.iloc[idx].cls_id
        word_1 = self.df.iloc[idx].word_1
        word_2 = self.df.iloc[idx].word_2
        word_3 = self.df.iloc[idx].word_3
        word_4 = self.df.iloc[idx].word_4
        word_5 = self.df.iloc[idx].word_5

        return {"lang_embed": lang_embed, "cls_id": cls_id, "idx": idx,
                "word_1": word_1, "word_2": word_2, "word_3": word_3, "word_4": word_4, "word_5": word_5}


class Collator:
    """
    Custom collate function so that the text tokenizer can be called on a batch
    of text, which is then padded to the length of the longest sequence in the batch.
    """
    def __init__(self, txt_tokenizer):
        self.txt_tokenizer = txt_tokenizer
    def __call__(self, batch):
        word_1 = self.txt_tokenizer(text=[s["word_1"] for s in batch], return_tensors="pt",
                                    padding=True, truncation=True)
        word_2 = self.txt_tokenizer(text=[s["word_2"] for s in batch], return_tensors="pt",
                                    padding=True, truncation=True)
        word_3 = self.txt_tokenizer(text=[s["word_3"] for s in batch], return_tensors="pt",
                                    padding=True, truncation=True)
        word_4 = self.txt_tokenizer(text=[s["word_4"] for s in batch], return_tensors="pt",
                                    padding=True, truncation=True)
        word_5 = self.txt_tokenizer(text=[s["word_5"] for s in batch], return_tensors="pt",
                                    padding=True, truncation=True)

        lang_embed = torch.Tensor([s["lang_embed"] for s in batch])
        cls_id = torch.Tensor([s["cls_id"] for s in batch])
        idx = torch.Tensor([s["idx"] for s in batch])

        batched_data = {"word_1_input_ids": word_1["input_ids"],
                        "word_1_attention_mask": word_1["attention_mask"],
                        "word_2_input_ids": word_2["input_ids"],
                        "word_2_attention_mask": word_2["attention_mask"],
                        "word_3_input_ids": word_3["input_ids"],
                        "word_3_attention_mask": word_3["attention_mask"],
                        "word_4_input_ids": word_4["input_ids"],
                        "word_4_attention_mask": word_4["attention_mask"],
                        "word_5_input_ids": word_5["input_ids"],
                        "word_5_attention_mask": word_5["attention_mask"],
                        "lang_embed": lang_embed, "cls_id": cls_id, "idx": idx}

        return batched_data


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))

    def text_tokenization(self):
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
        elif self.args.text_model_id == "google/mt5-base" or self.args.text_model_id == "google/mt5-small":
            self.txt_tokenizer =tokenizer = T5Tokenizer.from_pretrained(self.args.text_model_id)
            if self.args.text_embedding == "eos":
                self.feat_token_id = self.txt_tokenizer.eos_token_id
            elif self.args.text_embedding == "bos":
                self.feat_token_id = self.txt_tokenizer.bos_token_id


class HighDimDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.text_tokenization()

        df_train = pd.read_csv(self.args.data_dir / self.args.train_csv)
        self.ds_train = HighDimDataset(df_train)

        df_val = pd.read_csv(self.args.data_dir / self.args.val_csv)
        self.ds_val = HighDimDataset(df_val)

        df_test = pd.read_csv(self.args.data_dir / self.args.test_csv)
        self.ds_test = HighDimDataset(df_test)

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


def load_text_encoder(args, device):
    if args.text_model_id == "bert-base-multilingual-cased":
        text_encoder = BertModel.from_pretrained(args.text_model_id).to(device)
    elif args.text_model_id == "xlm-roberta-base":
        text_encoder = XLMRobertaModel.from_pretrained(args.text_model_id).to(device)
    elif args.text_model_id == "google/mt5-base" or args.text_model_id == "google/mt5-small":
        text_encoder = MT5EncoderModel.from_pretrained(args.text_model_id).to(device)
    text_encoder.eval()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    return text_encoder


def save_representations(args, text_encoder, dl, split):
    save_dir = args.save_dir / split
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    text_reps = []
    lang_embed = []
    cls_id = []
    idx = []

    word_1 = []
    word_2 = []
    word_3 = []
    word_4 = []
    word_5 = []

    for ix, batch in enumerate(tqdm(dl)):
        keys_to_device = ["word_1_input_ids", "word_1_attention_mask",
                          "word_2_input_ids", "word_2_attention_mask",
                          "word_3_input_ids", "word_3_attention_mask",
                          "word_4_input_ids", "word_4_attention_mask",
                          "word_5_input_ids", "word_5_attention_mask"]
        batch = {k: v.to(device) if k in keys_to_device else v for k, v in batch.items()}

        x = text_encoder(input_ids=batch["word_1_input_ids"], attention_mask=batch["word_1_attention_mask"])
        x = x["last_hidden_state"] # (b, l, d)
        x = x[torch.arange(x.shape[0]), (batch["word_1_input_ids"] == args.feat_token_id).int().argmax(dim=-1)] # (b, d)
        x = x.cpu()
        word_1.append(x)

        x = text_encoder(input_ids=batch["word_2_input_ids"], attention_mask=batch["word_2_attention_mask"])
        x = x["last_hidden_state"] # (b, l, d)
        x = x[torch.arange(x.shape[0]), (batch["word_2_input_ids"] == args.feat_token_id).int().argmax(dim=-1)] # (b, d)
        x = x.cpu()
        word_2.append(x)

        x = text_encoder(input_ids=batch["word_3_input_ids"], attention_mask=batch["word_3_attention_mask"])
        x = x["last_hidden_state"] # (b, l, d)
        x = x[torch.arange(x.shape[0]), (batch["word_3_input_ids"] == args.feat_token_id).int().argmax(dim=-1)] # (b, d)
        x = x.cpu()
        word_3.append(x)

        x = text_encoder(input_ids=batch["word_4_input_ids"], attention_mask=batch["word_4_attention_mask"])
        x = x["last_hidden_state"] # (b, l, d)
        x = x[torch.arange(x.shape[0]), (batch["word_4_input_ids"] == args.feat_token_id).int().argmax(dim=-1)] # (b, d)
        x = x.cpu()
        word_4.append(x)

        x = text_encoder(input_ids=batch["word_5_input_ids"], attention_mask=batch["word_5_attention_mask"])
        x = x["last_hidden_state"] # (b, l, d)
        x = x[torch.arange(x.shape[0]), (batch["word_5_input_ids"] == args.feat_token_id).int().argmax(dim=-1)] # (b, d)
        x = x.cpu()
        word_5.append(x)

        lang_embed.append(batch["lang_embed"])
        cls_id.append(batch["cls_id"])
        idx.append(batch["idx"])

    lang_embed = torch.cat(lang_embed, dim=0)
    torch.save(lang_embed, save_dir / f'lang_embed_{split}.pt')

    cls_id = torch.cat(cls_id, dim=0)
    torch.save(cls_id, save_dir / f'cls_id_{split}.pt')

    idx = torch.cat(idx, dim=0)
    torch.save(idx, save_dir / f'idx_{split}.pt')

    word_1 = torch.cat(word_1, dim=0)
    torch.save(word_1, save_dir / f'word_1_{split}.pt')

    word_2 = torch.cat(word_2, dim=0)
    torch.save(word_2, save_dir / f'word_2_{split}.pt')

    word_3 = torch.cat(word_3, dim=0)
    torch.save(word_3, save_dir / f'word_3_{split}.pt')

    word_4 = torch.cat(word_4, dim=0)
    torch.save(word_4, save_dir / f'word_4_{split}.pt')

    word_5 = torch.cat(word_5, dim=0)
    torch.save(word_5, save_dir / f'word_5_{split}.pt')


if __name__ == '__main__':
    args = parse_args_save_representations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = load_text_encoder(args, device)

    dm = HighDimDataModule(args)
    dm.prepare_data()

    dm.setup(stage="fit")
    args.feat_token_id = dm.feat_token_id
    print(f"Saving train tensors...")
    save_representations(args, text_encoder, dm.train_dataloader(), "train")
    print(f"Saving val tensors...")
    save_representations(args, text_encoder, dm.val_dataloader(), "val")

    dm.setup(stage="test")
    print(f"Saving test tensors...")
    save_representations(args, text_encoder, dm.test_dataloader(), "test")