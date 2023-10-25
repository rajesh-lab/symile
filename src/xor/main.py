"""
Experiment to demonstrate performance of SYMILE on synthetic datasets.
"""
from datetime import datetime
import os

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import parse_args
from datasets import XORDataModule
from models import XORModule


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'

    args = parse_args()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    if not os.path.exists(args.ckpt_save_dir):
        os.makedirs(args.ckpt_save_dir)
    save_dir = args.ckpt_save_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    setattr(args, "save_dir", save_dir)

    wandb_run_name = args.wandb_run_name if args.wandb_run_name != "None" \
        else f"{args.loss_fn}_{args.evaluation}"
    if args.wandb:
        logger = WandbLogger(project="symile", log_model="all",
                             name=wandb_run_name, save_dir=args.ckpt_save_dir)
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename="{epoch}-{val_loss:.2f}",
                                          mode="min",
                                          monitor="val_loss")
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=args.enable_progress_bar,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0
    )

    dm = XORDataModule(args)

    trainer.fit(XORModule(**vars(args)), datamodule=dm)
    trainer.test(ckpt_path="best", datamodule=dm)