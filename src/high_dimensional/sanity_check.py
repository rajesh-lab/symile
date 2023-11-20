from argparse import Namespace
from datetime import datetime
import os

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import BertModel, CLIPVisionModel, WhisperModel, XLMRobertaModel
from transformers import BertTokenizer, CLIPImageProcessor, \
                         WhisperFeatureExtractor, XLMRobertaTokenizer
from sklearn.model_selection import train_test_split

from args import parse_args_pretrain
from datasets import BaseDataModule


class SymilePrecomputedDataset(Dataset):
    def __init__(self, dataset_dir, split):
        self.audio = torch.load(dataset_dir / f"audio_{split}.pt")
        self.image = torch.load(dataset_dir / f"image_{split}.pt")
        self.text = torch.load(dataset_dir / f"text_{split}.pt")
        self.language = torch.load(dataset_dir / f"language_{split}.pt")
        self.object = torch.load(dataset_dir / f"object_{split}.pt")

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        return {"audio": self.audio[idx],
                "image": self.image[idx],
                "text": self.text[idx],
                "language": self.language[idx],
                "object": self.object[idx]}


class PretrainPrecomputedDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.text_tokenization()

        self.ds_train = SymilePrecomputedDataset(self.args.precomputed_rep_dir, "train")
        self.ds_val = SymilePrecomputedDataset(self.args.precomputed_rep_dir, "val")
        self.ds_test = SymilePrecomputedDataset(self.args.precomputed_rep_dir, "test")

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.args.batch_sz,
                          shuffle=True,
                          num_workers=self.num_workers,
                          drop_last=False)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          drop_last=False)


class SanityCheckModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # classify language from all text
        self.language_from_text_linear = nn.Sequential(
            nn.Linear(768, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, 5, bias=False)
        )

        # classify language from audio from template 1
        self.language_from_audio_linear = nn.Sequential(
            nn.Linear(768, 100, bias=False),
            nn.ReLU(),
            nn.Linear(100, 5, bias=False)
        )

        # # classify objects from images from template 1
        # self.object_from_image_linear = nn.Sequential(
        #     nn.Linear(768, 50, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(50, 100, bias=False)
        # )

        # # classify objects from text from template 1
        # self.object_from_text_linear = nn.Sequential(
        #     nn.Linear(768, 50, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(50, 100, bias=False)
        # )

        # classify objects from images from template 1
        self.object_from_image_linear = nn.Linear(768, 100, bias=False)

        # classify objects from text from template 1
        self.object_from_text_linear = nn.Linear(768, 100, bias=False)

    def forward(self, x):
        if self.args.classify == "language_from_text":
            logits = self.language_from_text_linear(x["text"])
        elif self.args.classify == "language_from_audio":
            logits = self.language_from_audio_linear(x["audio"])
        elif self.args.classify == "object_from_image":
            logits = self.object_from_image_linear(x["image"])
        elif self.args.classify == "object_from_text":
            logits = self.object_from_text_linear(x["text"])
        return logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def _shared_step(self, batch, batch_idx):
        logits = self(batch)

        if self.args.classify == "language_from_text" or self.args.classify == "language_from_audio":
            labels = batch["language"].type(torch.LongTensor).to(self.device)
        elif self.args.classify == "object_from_image" or self.args.classify == "object_from_text":
            labels = batch["object"].type(torch.LongTensor).to(self.device)

        loss = F.cross_entropy(logits, labels)

        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch, batch_idx)

        self.log_dict({"loss_train": loss},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch, batch_idx)
        pred = torch.argmax(logits, dim=1)
        acc = (torch.sum(labels == pred) / len(labels)).item()

        self.log_dict({"acc_val": acc},
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log_dict({"loss_val": loss},
                      on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch, batch_idx)
        pred = torch.argmax(logits, dim=1)
        acc = (torch.sum(labels == pred) / len(labels)).item()

        self.log_dict({"acc_test": acc},
                         on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return acc


def main(args, trainer):
    dm = PretrainPrecomputedDataModule(args)
    dm.setup(stage="fit")

    symile_model = SanityCheckModel(args)
    trainer.fit(symile_model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == '__main__':
    # SET THIS
    # classify language from all text
    # classify = "language_from_text"
    # classify objects from text from template 1
    classify = "object_from_text"
    # classify language from audio from template 1
    # classify = "language_from_audio"
    # classify objects from images from template 1
    # classify = "object_from_image"

    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'

    torch.set_float32_matmul_precision('medium')

    args = parse_args_pretrain()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    if not os.path.exists(args.ckpt_save_dir):
        os.makedirs(args.ckpt_save_dir)
    save_dir = args.ckpt_save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    setattr(args, "save_dir", save_dir)
    setattr(args, "classify", classify)

    if args.wandb:
        logger = WandbLogger(project="symile", log_model="all", save_dir=args.ckpt_save_dir)
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename="{epoch}-{loss_val:.2f}",
                                          mode="min",
                                          monitor="loss_val")
    profiler = None if args.profiler == "none" else args.profiler

    trainer = Trainer(
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        profiler=profiler
    )

    main(args, trainer)