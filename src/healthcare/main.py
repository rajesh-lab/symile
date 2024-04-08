from datetime import datetime
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import parse_args_main
from datasets import HighDimDataModule
from models import SSLModel


def main(args, trainer):
    dm = HighDimDataModule(args)

    if args.load_from_ckpt == "None":
        print("Training from scratch!")
        model = SSLModel(**vars(args))
    else:
        print("Loading checkpoint from ", args.load_from_ckpt)
        model = SSLModel.load_from_checkpoint(args.load_from_ckpt)

    trainer.fit(model, datamodule=dm)

    trainer.test(ckpt_path="best", datamodule=dm)


def test(args, trainer):
    dm = HighDimDataModule(args)

    print("Loading checkpoint from ", args.load_from_ckpt)
    model = SSLModel.load_from_checkpoint(args.load_from_ckpt)

    # override model args
    model.args.data_dir = args.data_dir
    model.args.save_dir = args.save_dir

    model.eval()
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    start = time.time()

    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'
    else:
        if os.getcwd().split("/")[3] == "as16583":
            os.environ['WANDB_CACHE_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
            os.environ['WANDB_CONFIG_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
            os.environ['WANDB_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
            os.environ['WANDB_DATA_DIR'] = '/gpfs/scratch/as16583/python_cache/wandb/'
        else:
            os.environ['WANDB_CACHE_DIR'] = '/gpfs/scratch/pulia01/adriel/wandb/'
            os.environ['WANDB_CONFIG_DIR'] = '/gpfs/scratch/pulia01/adriel/wandb/'
            os.environ['WANDB_DIR'] = '/gpfs/scratch/pulia01/adriel/wandb/'
            os.environ['WANDB_DATA_DIR'] = '/gpfs/scratch/pulia01/adriel/wandb/'

    args = parse_args_main()

    randint = random.randint(0, 9999) # to reduce chance of directory name collision when scripts are run in parallel
    save_dir = args.ckpt_save_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{randint:04d}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setattr(args, "save_dir", save_dir)
    print("\nSaving to: ", save_dir)

    if args.wandb:
        logger = WandbLogger(project="symile", log_model=False, save_dir=args.ckpt_save_dir)
    else:
        logger = False

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    val_loss_checkpoint = ModelCheckpoint(dirpath=save_dir,
                                          filename="best_val_loss_{epoch}-{val_loss:.2f}",
                                          mode="min",
                                          monitor="val_loss")

    general_checkpoint = ModelCheckpoint(dirpath=save_dir,
                                         filename="{epoch}-{val_loss:.2f}",
                                         every_n_epochs=args.check_val_every_n_epoch,
                                         save_top_k=-1)

    trainer = Trainer(
        callbacks=[val_loss_checkpoint, general_checkpoint],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=1,
        profiler=None
    )

    if args.step == "train":
        main(args, trainer)
    elif args.step == "test":
        test(args, trainer)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")