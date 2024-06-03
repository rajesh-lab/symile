"""Entry-point script to train using Symile or pairwise CLIP."""

from datetime import datetime
import importlib
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import parse_args_main
import datasets


def get_data_module(args):
    """
    Returns the appropriate DataModule based on the experiment.
    """
    if args.experiment == "symile_m3":
        dm = datasets.SymileM3DataModule
    elif args.experiment == "cxr_prediction":
        dm = datasets.CXRPredictionDataModule
    else:
        raise ValueError("Unsupported experiment name specified.")

    return dm(args)


def get_model_module(args):
    """
    Imports and returns the appropriate model module based on the experiment.
    """
    if args.experiment == "symile_m3":
        module = importlib.import_module("models.symile_m3_model")
        ModelClass = getattr(module, "SymileM3Model")
    elif args.experiment == "cxr_prediction":
        module = importlib.import_module("models.cxr_prediction_model")
        ModelClass = getattr(module, "CXRPredictionModel")
    else:
        raise ValueError("Unsupported experiment name specified.")

    return ModelClass(**vars(args))


def main(args, trainer):
    dm = get_data_module(args)

    if args.missingness:
        setattr(args, "tokenizer_len", dm.tokenizer_len)

    model = get_model_module(args)

    if args.load_from_ckpt == "None":
        print("Training model from scratch!")
        trainer.fit(model, datamodule=dm)
    else:
        print("Loading checkpoint from ", args.load_from_ckpt)
        trainer.fit(model, datamodule=dm, ckpt_path=args.load_from_ckpt)


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
        logger = WandbLogger(project="symile", log_model=False,
                             save_dir=args.ckpt_save_dir, id=args.wandb_run_id)
    else:
        logger = False

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    checkpoint_callback = ModelCheckpoint(dirpath=save_dir,
                                          filename="{epoch}-{val_loss:.4f}-{val_acc:.4f}",
                                          every_n_epochs=args.check_val_every_n_epoch,
                                          save_top_k=-1)

    trainer = Trainer(
        callbacks=checkpoint_callback,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        log_every_n_steps=1,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        profiler=None
    )

    main(args, trainer)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")