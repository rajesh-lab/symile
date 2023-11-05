"""
Experiment to demonstrate performance of SYMILE on synthetic datasets.
"""
from datetime import datetime
import os

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import plotly.express as px
from pytorch_lightning.loggers import WandbLogger
import torch

from args import parse_args
from datasets import SyntheticDataModule
from informations import best_accuracy, mutual_informations
from models import SyntheticModule
from utils import get_test_distribution


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'

    args = parse_args()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.ckpt_save_dir / datetime_now

    # compute mutual informations, total correlation, and best accuracy
    # for all values of i_p
    mi_results = {"i_p": [], "value": [], "type": []}
    acc_results = {"i_p": [], "loss_fn": [], "acc": []}
    loss_results = {"i_p": [], "type": [], "value": []}
    for i_p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        # mutual informations and total correlation
        mi = mutual_informations(i_p, args.d_v)
        mi["total_corr"] = mi["mi_a_c"] + mi["mi_b_c"] + mi["mi_a_b_given_c"]
        for k, v in mi.items():
            mi_results["i_p"].append(i_p)
            mi_results["type"].append(k)
            mi_results["value"].append(v)

        # accuracy of best predictor
        acc_results["i_p"].append(i_p)
        acc_results["loss_fn"].append("best_predictor")
        acc_results["acc"].append(best_accuracy(i_p, args.d_v))

        loss_results["i_p"].append(i_p)
        loss_results["type"].append("total_corr")
        loss_results["value"].append(mi["total_corr"])

    for loss_fn in ["symile", "pairwise_infonce"]:
        for i_p in [0.4]:
        # for i_p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_save_dir = save_dir / datetime_now
            if not os.path.exists(run_save_dir):
                os.makedirs(run_save_dir)

            setattr(args, "i_p", i_p)
            setattr(args, "loss_fn", loss_fn)
            setattr(args, "run_save_dir", run_save_dir)

            wandb_run_name = args.wandb_run_name if args.wandb_run_name != None \
                else f"{args.loss_fn}_{args.evaluation}_{datetime_now}"
            if args.wandb:
                logger = WandbLogger(project="symile", log_model="all",
                                    name=wandb_run_name, save_dir=run_save_dir)
            else:
                logger = False

            checkpoint_callback = ModelCheckpoint(dirpath=run_save_dir,
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

            print(f"\n***** Running {loss_fn} with i_p = {i_p}... *****\n")

            dm = SyntheticDataModule(args)
            model = SyntheticModule(**vars(args))

            trainer.fit(model, datamodule=dm)
            test_res = trainer.test(ckpt_path="best", datamodule=dm)[0]

            acc_results["i_p"].append(i_p)
            acc_results["loss_fn"].append(loss_fn)
            acc_results["acc"].append(test_res["mean_acc"])

            loss_results["i_p"].append(i_p)
            loss_results["type"].append(f"test_loss_{loss_fn}")
            loss_results["value"].append(test_res["test_loss_epoch"])
            loss_results["i_p"].append(i_p)
            loss_results["type"].append(f"log_n_minus_1_{loss_fn}")
            loss_results["value"].append(test_res["test_log_n_minus_1"])

            test_dist_df = get_test_distribution(dm)

            if args.wandb:
                logger.experiment.finish()

    mi_df = pd.DataFrame(mi_results)
    mi_df.to_csv(save_dir / "mi.csv", index=False)
    fig = px.line(mi_df, x="i_p", y="value", color="type")
    fig.write_image(save_dir / "mi.png")

    acc_df = pd.DataFrame(acc_results)
    acc_df.to_csv(save_dir / "acc.csv", index=False)
    fig = px.line(acc_df, x="i_p", y="acc", color="loss_fn")
    fig.write_image(save_dir / "acc.png")

    loss_df = pd.DataFrame(loss_results)
    loss_df.to_csv(save_dir / "loss.csv", index=False)
    fig = px.line(loss_df, x="i_p", y="value", color="type")
    fig.write_image(save_dir / "loss.png")

    test_dist_df.to_csv(save_dir / "test_dist.csv", index=False)
    fig = px.line(test_dist_df, x="value", y="prob", color="type")
    fig.write_image(save_dir / "test_dist.png")