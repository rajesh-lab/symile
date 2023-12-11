import os

import lightning.pytorch as pl
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, XLMRobertaTokenizer, \
                         BertModel, XLMRobertaModel
from tqdm import tqdm

from args import parse_args_save_representations


class HighDimDataset(Dataset):
    def __init__(self, df):
        self.df = df

        langs = sorted(df["lang"].unique())
        self.lang_embeddings = {value: idx for idx, value in enumerate(langs)}

        classes = sorted(df["cls"].unique())
        self.img_classes = {value: idx for idx, value in enumerate(classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        lang_embed = self.lang_embeddings[self.df.iloc[idx].lang]
        img_cls = self.img_classes[self.df.iloc[idx].cls]
        text = self.df.iloc[idx].text

        return {"lang_embed": lang_embed, "img_cls": img_cls, "text": text,
                "idx": idx}


class Collator:
    """
    Custom collate function so that the text tokenizer can be called on a batch
    of text, which is then padded to the length of the longest sequence in the batch.
    """
    def __init__(self, txt_tokenizer):
        self.txt_tokenizer = txt_tokenizer
    def __call__(self, batch):
        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding=True, truncation=True)

        lang_embed = torch.Tensor([s["lang_embed"] for s in batch])
        img_cls = torch.Tensor([s["img_cls"] for s in batch])
        idx = torch.Tensor([s["idx"] for s in batch])

        batched_data = {"text_input_ids": text["input_ids"],
                        "text_attention_mask": text["attention_mask"],
                        "lang_embed": lang_embed, "img_cls": img_cls, "idx": idx}

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
    img_cls = []
    idx = []

    for ix, batch in enumerate(tqdm(dl)):
        keys_to_device = ["text_input_ids", "text_attention_mask"]
        batch = {k: v.to(device) if k in keys_to_device else v for k, v in batch.items()}

        # text encoder
        x = text_encoder(input_ids=batch["text_input_ids"],
                         attention_mask=batch["text_attention_mask"])
        x = x["last_hidden_state"] # (b, l, d)
        x = x[torch.arange(x.shape[0]),
              (batch["text_input_ids"] == args.feat_token_id).int().argmax(dim=-1)] # (b, d)
        x = x.cpu()
        text_reps.append(x)

        lang_embed.append(batch["lang_embed"])
        img_cls.append(batch["img_cls"])
        idx.append(batch["idx"])

    text_reps = torch.cat(text_reps, dim=0)
    torch.save(text_reps, save_dir / f'text_{split}.pt')

    lang_embed = torch.cat(lang_embed, dim=0)
    torch.save(lang_embed, save_dir / f'lang_embed_{split}.pt')

    img_cls = torch.cat(img_cls, dim=0)
    torch.save(img_cls, save_dir / f'img_cls_{split}.pt')

    idx = torch.cat(idx, dim=0)
    torch.save(idx, save_dir / f'idx_{split}.pt')


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