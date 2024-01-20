from argparse import Namespace
from datetime import datetime
import os

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor

from args import parse_args_main
from src.utils import l2_normalize


class HighDimDataset(Dataset):
    def __init__(self, df, data_dir, split):
        self.df = df
        self.split_dir = data_dir / f"{split}"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = np.load(self.split_dir / f"image/{self.df.iloc[idx].image_filename}.npy")
        cls_id = self.df.iloc[idx].cls_id

        return {"image": image,
                "cls_id": cls_id}


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # from max_num_worker_suggest in DataLoader docs
        self.num_workers = len(os.sched_getaffinity(0))


class HighDimDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
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
                          drop_last=self.args.drop_last)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.args.batch_sz_val,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.args.batch_sz_test,
                          num_workers=self.num_workers,
                          drop_last=self.args.drop_last)


class SSLModel(pl.LightningModule):
    def __init__(self, **args):
        super().__init__()
        self.save_hyperparameters()

        self.args = Namespace(**args)

        enc = CLIPVisionModel.from_pretrained(self.args.image_model_id)
        self.fc = nn.Linear(enc.config.hidden_size, self.args.d, bias=True)
        self.layer_norm = nn.LayerNorm(self.args.d)
        self.classifier = nn.Linear(self.args.d, self.args.num_classes, bias=True)

    def forward(self, x):
        r_i = self.fc(x["image"])
        r_i = self.layer_norm(r_i)
        r_i = self.classifier(r_i)
        return r_i

    def configure_optimizers(self):
        assert self.args.weight_decay == 0.0
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def _shared_step(self, batch, batch_idx):
        logits = self(batch) # (b, num_classes)
        labels = batch["cls_id"].long() # (b)

        pred = torch.argmax(logits, dim=1)
        acc = torch.sum(pred == labels) / len(labels)
        self.log("train_accuracy", acc, sync_dist=True, prog_bar=True)

        loss = nn.CrossEntropyLoss()
        return loss(logits, labels)

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)

        self.log("train_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        logits = self(batch) # (b, 1000)
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