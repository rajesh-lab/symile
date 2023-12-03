from datetime import datetime
import os
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch
import torch.nn as nn

from args import parse_args_main


class SSLModel(pl.LightningModule):
    def __init__(self, feat_token_id):
        super().__init__()
        self.feat_token_id = feat_token_id

        self.text_encoder = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.language_embedding = nn.Embedding(5, 768)
        self.fc1 = nn.Linear(10, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 5)

    def forward(self, x):
        txt_embed = self.text_encoder(input_ids=x["text_input_ids"],
                                      attention_mask=x["text_attention_mask"])
        txt_embed = txt_embed.last_hidden_state[:, 0, :]
        breakpoint()
        # take features from EOS or BOS embedding. x has shape (b, l, d).
        # argmax returns first index of feat_token_id in case pad_token_id is
        # equal to feat_token_id.
        txt_embed = txt_embed[torch.arange(txt_embed.shape[0]),
                  (x["text_input_ids"] == self.feat_token_id).int().argmax(dim=-1)]

        lang_embed = self.language_embedding(x["lang_embed"].to(torch.int))

        return r_a, r_i, r_t, self.logit_scale.exp()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1.0e-3)

    def _shared_step(self, batch, batch_idx):
        r_a, r_i, r_t, logit_scale_exp = self(batch)

        if self.args.normalize:
            r_a, r_i, r_t = l2_normalize([r_a, r_i, r_t])

        loss = self.loss_fn(r_a, r_i, r_t, logit_scale_exp)

        return loss, logit_scale_exp

    def training_step(self, batch, batch_idx):
        loss, logit_scale_exp = self._shared_step(batch, batch_idx)

        log_n_minus_1 = np.log(len(batch[list(batch.keys())[0]]) - 1)

        self.log_dict({"train_loss": loss, "logit_scale_exp": logit_scale_exp},
                      on_step=True, on_epoch=True, sync_dist=False, prog_bar=True)
        self.log("log_n_minus_1", log_n_minus_1,
                 on_step=False, on_epoch=True, sync_dist=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logit_scale_exp = self._shared_step(batch, batch_idx)

        log_n_minus_1 = np.log(len(batch[list(batch.keys())[0]]) - 1)

        self.log("val_loss", loss,
                 on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("log_n_minus_1", log_n_minus_1,
                 on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        return loss

    def test_step(self, batch, batch_idx):
        """
        The zeroshot task is to predict which r_i corresponds to a given r_a, r_t.
        """
        r_a, _, r_t, _ = self(batch)

        if self.args.normalize:
            r_a, r_t = l2_normalize([r_a, r_t])

        zeroshot_acc = self.zeroshot_accuracy(r_a, r_t, batch["idx"])

        self.log("test_accuracy", zeroshot_acc, sync_dist=True, prog_bar=True)


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