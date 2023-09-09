from datetime import datetime
import os
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

from args import parse_args_pretrain
from datasets import PretrainDataModule
from models import SymileModel


def pretrain(args, trainer, logger):
    dm = PretrainDataModule(args)
    dm.setup(stage="fit")
    args.feat_token_id = dm.feat_token_id

    symile_model = SymileModel(**vars(args))
    if args.wandb:
        logger.watch(symile_model)

    trainer.fit(symile_model, datamodule=dm)


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'

    args = parse_args_pretrain()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    save_dir = Path("./ckpts/pretrain")
    if args.wandb:
        logger = WandbLogger(project="symile", log_model="all", save_dir=save_dir)
    else:
        logger = False

    if logger:
        dirpath = save_dir / logger.experiment.id
    else:
        dirpath = save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")

    checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                          filename="{epoch}-{val_loss:.2f}",
                                          mode="min",
                                          monitor="val_loss")
    early_stopping_callback = EarlyStopping(monitor="val_loss",
                                            mode="min",
                                            patience=args.early_stopping_patience)
    # `ddp_find_unused_parameters_true` instead of `ddp` because error is thrown
    # when not all indices of nn.Embedding are used in a minibatch
    profiler = None if args.profiler == "none" else args.profiler
    strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        profiler=profiler,
        strategy=strategy
    )

    pretrain(args, trainer, logger)