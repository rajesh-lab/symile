from argparse import Namespace
from datetime import datetime
import os

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from transformers import BertTokenizer, XLMRobertaTokenizer, \
                         BertModel, XLMRobertaModel, \
                         MT5EncoderModel, T5Tokenizer, \
                         WhisperModel

from args import parse_args_main
from src.high_dim.constants import LANGUAGES


class HighDimDataset(Dataset):
    def __init__(self, df, data_dir, split):
        self.df = df
        self.split_dir = data_dir / f"{split}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio = np.load(self.split_dir / f"audio/{self.df.iloc[idx].audio_filename}.npy")
        lang = self.df.iloc[idx].lang
        lang_id = LANGUAGES[lang]

        text = self.df.iloc[idx].text

        cls_id = self.df.iloc[idx].cls_id

        return {"audio": audio,
                "lang": lang,
                "lang_id": lang_id,
                "text": text,
                "cls_id": cls_id}


class Collator:
    """
    Custom collate function so that the text tokenizer can be called on a batch
    of text, which is then padded to the length of the longest sequence in the batch.
    """
    def __init__(self, txt_tokenizer):
        self.txt_tokenizer = txt_tokenizer
    def __call__(self, batch):
        audio = torch.tensor(np.stack([s["audio"] for s in batch]))

        text_list = [s["text"] for s in batch]
        text = self.txt_tokenizer(text=text_list, return_tensors="pt",
                                  padding=True, truncation=True)

        lang = [s["lang"] for s in batch]
        lang_id = torch.tensor([s["lang_id"] for s in batch])
        cls_id = torch.tensor([s["cls_id"] for s in batch])

        batched_data = {
            "audio": audio,
            "text": text,
            "lang": lang,
            "lang_id": lang_id,
            "cls_id": cls_id}

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


class AudioEncoder(nn.Module):
    def __init__(self, model_id, d):
        super().__init__()
        encoder = WhisperModel.from_pretrained(model_id).encoder

        self.fc = nn.Linear(encoder.config.hidden_size, d, bias=True)

    def forward(self, x):
        x = self.fc(x)
        x = x.mean(dim=1)
        return x

class TextEncoder(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        if model_id == "bert-base-multilingual-cased":
            self.encoder = BertModel.from_pretrained(model_id)
        elif model_id == "xlm-roberta-base":
            self.encoder = XLMRobertaModel.from_pretrained(model_id)

    def forward(self, x):
        x = self.encoder(**x)
        x = x[1] # get pooled output
        return x

class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)

        self.audio_encoder = AudioEncoder(self.args.audio_model_id, self.args.d)
        self.audio_layer_norm = nn.LayerNorm(self.args.d)

        self.text_encoder = TextEncoder(self.args.text_model_id)
        self.text_layer_norm = nn.LayerNorm(self.args.d)

        self.classifier = nn.Linear(self.args.d, self.args.num_classes, bias=True)

    def forward(self, x):
        r_a = self.audio_encoder(x["audio"])
        r_a = self.audio_layer_norm(r_a)

        r_t = self.text_encoder(x["text"])
        r_t = self.text_layer_norm(r_t)

        input_tensor = r_a * r_t

        x = self.classifier(input_tensor)
        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def _shared_step(self, batch, batch_idx):
        logits = self(batch) # (b, num_classes)
        labels = batch["cls_id"].long() # (b)

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == labels) / len(labels)

        loss = nn.CrossEntropyLoss()
        return loss(logits, labels), acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)

        self.log("train_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        self.log("train_accuracy", acc, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch, batch_idx)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_accuracy", acc, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch) # (b, num_classes)
        labels = batch["cls_id"].long() # (b)

        pred = torch.argmax(logits, dim=1)

        acc = torch.sum(pred == labels) / len(labels)

        self.log("test_accuracy", acc, sync_dist=True, prog_bar=True)


def main(args, trainer):
    dm = HighDimDataModule(args)
    dm.setup(stage="fit")

    model = SSLModel(**vars(args))
    trainer.fit(model, datamodule=dm)

    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == '__main__':
    os.environ['WANDB_CACHE_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
    os.environ['WANDB_CONFIG_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
    os.environ['WANDB_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
    os.environ['WANDB_DATA_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'

    args = parse_args_main()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    save_dir = args.ckpt_save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(args.ckpt_save_dir):
        os.makedirs(args.ckpt_save_dir)
    os.mkdir(save_dir)
    setattr(args, "save_dir", save_dir)
    print("\nSaving to: ", save_dir)

    if args.wandb:
        logger = WandbLogger(project="symile", log_model="all", save_dir=args.ckpt_save_dir)
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          every_n_epochs = 1,
                                          filename="{epoch}-{val_loss:.2f}",
                                          mode="min",
                                          monitor="val_loss")
    early_stopping_callback = EarlyStopping(monitor="val_loss",
                                            mode="min",
                                            patience=args.early_stopping_patience)
    if args.early_stopping:
        callbacks = [checkpoint_callback, early_stopping_callback]
    else:
        callbacks = [checkpoint_callback]

    trainer = Trainer(
        callbacks=callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0
    )

    main(args, trainer)