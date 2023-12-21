"""
Experiments to demonstrate performance of SYMILE on binary data.
"""
from datetime import datetime
import os

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import pandas as pd
import plotly.express as px
from pytorch_lightning.loggers import WandbLogger

from args import parse_args
from datasets import BinaryDataModule
from models import BinaryModule
from utils import likelihood_ratios, mutual_informations, \
                  save_likelihood_ratio_vs_score, save_test_distribution


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'

    args = parse_args()

    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.ckpt_save_dir / datetime_now
    os.makedirs(save_dir)
    print(f"save_dir is {save_dir}")

    print("computing mutual informations and total correlation for all values of i_p...")
    acc_results = {"i_p": [], "loss_fn": [], "acc": []}
    mi_results = {"i_p": [], "value": [], "type": []}
    loss_results = {"i_p": [], "type": [], "value": []}

    if args.d_v <= 5:
        for i_p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            # mutual informations and total correlation
            mi = mutual_informations(args.d_v, i_p)
            mi["total_corr"] = mi["mi_a_c"] + mi["mi_b_c"] + mi["mi_a_b_given_c"]

            for k, v in mi.items():
                mi_results["i_p"].append(i_p)
                mi_results["type"].append(k)
                mi_results["value"].append(v)

            if args.save_loss_results:
                loss_results["i_p"].append(i_p)
                loss_results["type"].append("total_corr")
                loss_results["value"].append(mi["total_corr"])

    if args.save_likelihood_ratios:
        print("calculating true likelihood ratio p(a,b,c)/p(a)p(b)p(c) for each i_p...")
        lr_data = likelihood_ratios(args.d_v)

    print("training...")
    for loss_fn in ["symile", "clip"]:
        for i_p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            if args.use_seed:
                seed_everything(args.seed, workers=True)

            i_p_dir = save_dir / f"i_p_{i_p}"
            if not os.path.exists(i_p_dir):
                os.mkdir(i_p_dir)

            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_save_dir = save_dir / datetime_now
            if not os.path.exists(run_save_dir):
                os.makedirs(run_save_dir)

            setattr(args, "i_p", i_p)
            setattr(args, "loss_fn", loss_fn)
            setattr(args, "run_save_dir", run_save_dir)

            wandb_run_name = args.wandb_run_name if args.wandb_run_name != None \
                else f"{args.loss_fn}_{datetime_now}"
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

            dm = BinaryDataModule(args)
            model = BinaryModule(**vars(args))

            trainer.fit(model, datamodule=dm)
            test_res = trainer.test(ckpt_path="best", datamodule=dm)[0]

            acc_results["i_p"].append(i_p)
            acc_results["loss_fn"].append(loss_fn)
            acc_results["acc"].append(test_res["mean_acc"])

            if args.d_v <= 5 and args.save_loss_results:
                loss_results["i_p"].append(i_p)
                loss_results["type"].append(f"test_loss_{loss_fn}")
                loss_results["value"].append(test_res["test_loss_epoch"])
                loss_results["i_p"].append(i_p)
                loss_results["type"].append(f"log_n_minus_1_{loss_fn}")
                loss_results["value"].append(test_res["test_log_n_minus_1"])

            if args.save_likelihood_ratios:
                save_test_distribution(dm, i_p_dir, loss_fn, i_p)
                save_likelihood_ratio_vs_score(i_p, loss_fn, model, lr_data[i_p],
                                               i_p_dir, dim=args.d_v)

            if args.wandb:
                logger.experiment.finish()

    acc_df = pd.DataFrame(acc_results)
    acc_df.to_csv(save_dir / "acc.csv", index=False)
    fig = px.line(acc_df, x="i_p", y="acc", color="loss_fn")
    fig.write_image(save_dir / "acc.png")

    if args.d_v <= 5:
        mi_df = pd.DataFrame(mi_results)
        mi_df.to_csv(save_dir / "mi.csv", index=False)
        fig = px.line(mi_df, x="i_p", y="value", color="type")
        fig.write_image(save_dir / "mi.png")

    if args.save_loss_results:
        loss_df = pd.DataFrame(loss_results)
        loss_df.to_csv(save_dir / "loss.csv", index=False)
        fig = px.line(loss_df, x="i_p", y="value", color="type")
        fig.write_image(save_dir / "loss.png")