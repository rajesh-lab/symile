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

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from args import parse_args_test
from datasets import SupportClfDataModule
from models import SupportClfModel


def test_zeroshot(args, trainer, logger):
    model = SupportClfModel(**vars(args))


def test_support(args, trainer, logger):
    model = SupportClfModel(**vars(args))
    if args.wandb:
        logger.watch(model)

    # get arguments from pre-trained model
    setattr(args, "audio_model_id", model.model.args.audio_model_id)
    setattr(args, "image_model_id", model.model.args.image_model_id)
    setattr(args, "text_model_id", model.model.args.text_model_id)
    setattr(args, "text_embedding", model.model.args.text_embedding)

    dm = SupportClfDataModule(args)
    trainer.fit(model, datamodule=dm)
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'

    args = parse_args_test()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    logger = WandbLogger(project="symile", log_model="all") if args.wandb else None
    if logger:
        dirpath = f"./ckpts/{args.evaluation}/{logger.experiment.id}"
    else:
        dirpath = f"./ckpts/{args.evaluation}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                          filename="{epoch}-{val_loss:.2f}",
                                          mode="min",
                                          monitor="val_loss")
    early_stopping_callback = EarlyStopping(monitor="val_loss",
                                            mode="min",
                                            patience=args.early_stopping_patience)
    logger = WandbLogger(project="symile", log_model="all") if args.wandb else None
    # TODO: make sure this works with multiple GPUs!!!!

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0
    )

    if args.evaluation == "zeroshot":
        test_zeroshot(args, trainer, logger)
    elif args.evaluation == "support":
        test_support(args, trainer, logger)