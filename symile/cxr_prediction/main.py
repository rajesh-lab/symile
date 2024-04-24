from datetime import datetime
import json
from json import JSONEncoder
import os
from pathlib import Path
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from args import parse_args_main
from datasets import CXRPredictionDataModule
from models import SSLModel


class PathToStrEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return JSONEncoder.default(self, obj)


class LoggerCallback(Callback):
    def __init__(self, args):
        self.args = vars(args)
        self.run_info = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.logged_metrics
        metrics = {key: metrics[key].item() for key in metrics if key != "val_loss_step"}

        metrics["epoch"] = trainer.current_epoch

        self.run_info.setdefault("validation_metrics", []).append(metrics)

    def on_train_end(self, trainer, pl_module):
        self.run_info["args"] = self.args

        try:
            self.run_info["wandb"] = trainer.logger.experiment.url
        except AttributeError:
            self.run_info["wandb"] = None

        with open(self.args["save_dir"] / "run_info.json", "w") as f:
            json.dump(self.run_info, f, indent=4, cls=PathToStrEncoder)


def main(args, trainer):
    dm = CXRPredictionDataModule(args)

    if args.load_from_ckpt == "None":
        print("Training from scratch!")
        model = SSLModel(**vars(args))
    else:
        print("Loading checkpoint from ", args.load_from_ckpt)
        model = SSLModel.load_from_checkpoint(args.load_from_ckpt)

    trainer.fit(model, datamodule=dm)


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

    checkpoint = ModelCheckpoint(dirpath=save_dir,
                                 filename="{epoch}-{val_loss:.4f}",
                                 every_n_epochs=args.check_val_every_n_epoch,
                                 save_top_k=-1)

    logger_callback = LoggerCallback(args)

    trainer = Trainer(
        callbacks=[checkpoint, logger_callback],
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