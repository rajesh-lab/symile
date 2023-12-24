"""
predict class using text embedding and a language embedding
"""
from argparse import Namespace
from datetime import datetime
import os

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from args import parse_args_main
from save_representations import BaseDataModule, HighDimDataModule


class HighDimPrecomputedDataset(Dataset):
    def __init__(self, dataset_dir, split):
        split_dir = dataset_dir / f"{split}"
        self.text = torch.load(split_dir / f"text_{split}.pt")
        self.lang_embed = torch.load(split_dir / f"lang_embed_{split}.pt")
        self.img_cls = torch.load(split_dir / f"img_cls_{split}.pt")
        self.idx = torch.load(split_dir / f"idx_{split}.pt")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {"text": self.text[idx],
                "lang_embed": self.lang_embed[idx],
                "img_cls": self.img_cls[idx],
                "idx": self.idx[idx]}


class HighDimPrecomputedDataModule(BaseDataModule):
    def __init__(self, args):
        super().__init__(args)

    def setup(self, stage):
        self.text_tokenization()

        self.ds_train = HighDimPrecomputedDataset(self.args.precomputed_rep_dir, "train")
        self.ds_val = HighDimPrecomputedDataset(self.args.precomputed_rep_dir, "val")
        self.ds_test = HighDimPrecomputedDataset(self.args.precomputed_rep_dir, "test")

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

        self.lang_embedding = nn.Embedding(5, self.args.lang_embed_d).to(self.device)
        self.fc1 = nn.Linear(self.args.d + self.args.lang_embed_d, self.args.hidden_layer_d)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.args.hidden_layer_d, 5)

    def forward(self, x):
        if self.args.use_precomputed_representations:
            # x["text"] has shape (b, d); x["lang_embed"] has shape (b)
            lang_embed = self.lang_embedding(x["lang_embed"].to(torch.int))

            input_tensor = torch.cat((x["text"], lang_embed), dim=1)

            x = self.fc1(input_tensor)
            x = self.relu(x)
            x = self.fc2(x)
        else:
            ValueError("Not implemented.")

        return x

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.lr,
                                 weight_decay=self.args.weight_decay)

    def _shared_step(self, batch, batch_idx):
        logits = self(batch) # (b, 5)
        labels = batch["img_cls"].long() # (b)

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
        logits = self(batch) # (b, 5)
        labels = batch["img_cls"].long() # (b)

        pred = torch.argmax(logits, dim=1)

        acc = torch.sum(pred == labels) / len(labels)

        self.log("test_accuracy", acc, sync_dist=True, prog_bar=True)


def main(args, trainer):
    if args.use_precomputed_representations:
        dm = HighDimPrecomputedDataModule(args)
    else:
        dm = HighDimDataModule(args)
    dm.setup(stage="fit")
    args.feat_token_id = dm.feat_token_id

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