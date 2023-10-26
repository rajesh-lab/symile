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


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'
    # utilize a100s to trade-off precision for performance
    if torch.cuda.get_device_name() == "NVIDIA A100 80GB PCIe":
        torch.set_float32_matmul_precision('medium')

    args = parse_args()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.ckpt_save_dir / datetime_now

    results = {"loss_fn": [], "i_p": [], "acc": []}

    for loss_fn in ["symile", "pairwise_infonce"]:
        for i_p in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            mi = mutual_informations(i_p)
            best_acc = best_accuracy(i_p)

            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_save_dir = save_dir / datetime_now
            if not os.path.exists(run_save_dir):
                os.makedirs(run_save_dir)

            setattr(args, "i_p", i_p)
            setattr(args, "loss_fn", loss_fn)
            setattr(args, "mi_a_c", mi["mi_a_c"])
            setattr(args, "mi_b_c", mi["mi_b_c"])
            setattr(args, "mi_a_b_given_c", mi["mi_a_b_given_c"])
            setattr(args, "best_acc", best_acc)
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

            dm = SyntheticDataModule(args)

            print(f"\n***** Running {loss_fn} with i_p = {i_p}... *****\n")

            trainer.fit(SyntheticModule(**vars(args)), datamodule=dm)
            acc = trainer.test(ckpt_path="best", datamodule=dm)

            results["loss_fn"].append(loss_fn)
            results["i_p"].append(i_p)
            results["acc"].append(acc[0]["mean_acc"])

            if args.wandb:
                logger.experiment.finish()

    df = pd.DataFrame(results)
    df.to_csv(save_dir / "results.csv", index=False)
    fig = px.line(df, x="i_p", y="acc", color="loss_fn")
    fig.write_image(save_dir / "results.png")