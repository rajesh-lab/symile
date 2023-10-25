"""
Note that for support classification model fitting and testing are both in this
scipt: because we run trainer.test() directly after trainer.fit(),
trainer.test() automatically loads the best weights from training.
As a result, DDP should not be used when running this script (i.e. no more than
a single GPU should be used). During trainer.test(), it is recommended to use
Trainer(devices=1) to ensure each sample/batch gets evaluated exactly once.
Otherwise, multi-device settings use `DistributedSampler` that replicates some
samples to make sure all devices have same batch size in case of uneven inputs.
"""
from datetime import datetime
import os
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

from args import parse_args_test
from datasets import SupportClfDataModule, ZeroshotClfDataModule, \
                     SupportClfPrecomputedDataModule, ZeroshotClfPrecomputedDataModule
from models import SupportClfModel, ZeroshotClfModel


def pretrained_model_args(args, model):
    """add encoder arguments from pre-trained model"""
    setattr(args, "audio_model_id", model.model.args.audio_model_id)
    setattr(args, "image_model_id", model.model.args.image_model_id)
    setattr(args, "text_model_id", model.model.args.text_model_id)
    setattr(args, "text_embedding", model.model.args.text_embedding)
    return args


def test_zeroshot(args, trainer, logger):
    model = ZeroshotClfModel(**vars(args))
    args = pretrained_model_args(args, model)

    if args.use_precomputed_representations:
        dm = ZeroshotClfPrecomputedDataModule(args)
    else:
        dm = ZeroshotClfDataModule(args)

    trainer.test(model, datamodule=dm)


def test_support(args, trainer, logger):
    model = SupportClfModel(**vars(args))
    args = pretrained_model_args(args, model)

    if args.use_precomputed_representations:
        dm = SupportClfPrecomputedDataModule(args)
    else:
        dm = SupportClfDataModule(args)

    trainer.fit(model, datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'
    else:
        os.environ['WANDB_CACHE_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
        os.environ['WANDB_CONFIG_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
        os.environ['WANDB_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
        os.environ['WANDB_DATA_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'

    torch.set_float32_matmul_precision('medium')

    args = parse_args_test()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    if not os.path.exists(args.ckpt_save_dir):
        os.makedirs(args.ckpt_save_dir)
    save_dir = args.ckpt_save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
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

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0
    )

    if args.evaluation == "zeroshot":
        test_zeroshot(args, trainer, logger)
    elif args.evaluation == "support":
        test_support(args, trainer, logger)