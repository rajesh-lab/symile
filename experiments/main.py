"""Entry-point script to train using Symile or pairwise CLIP."""

from datetime import datetime
import importlib
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
from pytorch_lightning.loggers import WandbLogger

from args import parse_args_main
import datasets


def create_save_directory(args):
    """
    Create a unique save directory using the current timestamp and a random integer
    between 0 and 9999 in order to reduce the chance of directory name collision when
    scripts are run in parallel.
    """
    randint = random.randint(0, 9999)
    save_dir = args.ckpt_save_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{randint:04d}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def get_data_module(args):
    """
    Returns the appropriate DataModule based on the experiment.
    """
    if args.experiment == "binary_xor":
        dm = datasets.BinaryXORDataModule
    elif args.experiment == "symile_m3":
        dm = datasets.SymileM3DataModule
    elif args.experiment == "symile_mimic":
        dm = datasets.SymileMIMICDataModule
    else:
        raise ValueError("Unsupported experiment name specified.")

    return dm(args)


def get_model_module(args):
    """
    Imports and returns the appropriate model module based on the experiment.
    """
    if args.experiment == "binary_xor":
        module = importlib.import_module("models.binary_xor_model")
        ModelClass = getattr(module, "BinaryXORModel")
    elif args.experiment == "symile_m3":
        module = importlib.import_module("models.symile_m3_model")
        ModelClass = getattr(module, "SymileM3Model")
    elif args.experiment == "symile_mimic":
        module = importlib.import_module("models.symile_mimic_model")
        ModelClass = getattr(module, "SymileMIMICModel")
    else:
        raise ValueError("Unsupported experiment name specified.")

    return ModelClass(**vars(args))


def binary_xor_main(args):
    results = {"p_hat": [], "loss_fn": [], "acc": []}

    for p_hat in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print(f"\n***** running p_hat = {p_hat}... *****\n")

        p_hat_save_dir = args.save_dir / f"p_hat_{p_hat:.1f}"

        setattr(args, "p_hat", p_hat)
        setattr(args, "p_hat_save_dir", p_hat_save_dir)

        if args.wandb:
            logger = WandbLogger(project="symile", log_model=False,
                                 save_dir=p_hat_save_dir)
        else:
            logger = False

        checkpoint_callback = ModelCheckpoint(dirpath=p_hat_save_dir,
                                              filename="{epoch}-{val_loss:.4f}",
                                              mode="min",
                                              monitor="val_loss")

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

        dm = get_data_module(args)

        model = get_model_module(args)

        trainer.fit(model, datamodule=dm)

        if args.bootstrap:
            for i in range(args.bootstrap_n):
                print(f"\nRunning bootstrap iteration {i+1}/{args.bootstrap_n} for p_hat = {p_hat}...")

                dm.resample_test_set()

                metrics = trainer.test(ckpt_path="best", datamodule=dm)[0]

                results["p_hat"].append(p_hat)
                results["loss_fn"].append(args.loss_fn)
                results["acc"].append(metrics["test_acc"])
        else:
            metrics = trainer.test(ckpt_path="best", datamodule=dm)[0]

            results["p_hat"].append(p_hat)
            results["loss_fn"].append(args.loss_fn)
            results["acc"].append(metrics["test_acc"])

        if args.wandb:
            logger.experiment.finish()

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.save_dir / "binary_xor_results.csv", index=False)


def main(args):
    if args.wandb:
        logger = WandbLogger(project="symile", log_model=False,
                             save_dir=args.ckpt_save_dir, id=args.wandb_run_id)
    else:
        logger = False

    checkpoint_callback = ModelCheckpoint(dirpath=args.save_dir,
                                          filename="{epoch}-{val_loss:.4f}",
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

    dm = get_data_module(args)

    if args.experiment == "symile_m3" and args.missingness:
        setattr(args, "tokenizer_len", dm.tokenizer_len)

    model = get_model_module(args)

    if args.ckpt_path == "None":
        print("Training model from scratch!")
        trainer.fit(model, datamodule=dm)
    else:
        print("Loading checkpoint from ", args.ckpt_path)
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)


if __name__ == '__main__':
    start = time.time()

    args = parse_args_main()

    save_dir = create_save_directory(args)
    setattr(args, "save_dir", save_dir)
    print("\nSaving to: ", save_dir)

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    if args.experiment == "binary_xor":
        binary_xor_main(args)
    elif args.experiment in ["symile_m3", "symile_mimic"]:
        main(args)
    else:
        raise ValueError("Unsupported experiment name specified.")

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")