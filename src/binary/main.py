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


def main(seed, save_dir, args):
    """
    Runs models first for symile, then for clip. For each of the two loss
    functions, runs for each value of p_hat.
    """
    print(f"using seed {seed}")

    res = {"seed": [], "p_hat": [], "loss_fn": [], "acc": []}

    for loss_fn in ["symile", "clip"]:

        for p_hat in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:

            seed_everything(seed, workers=True)

            datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_save_dir = save_dir / datetime_now
            if not os.path.exists(run_save_dir):
                os.makedirs(run_save_dir)

            setattr(args, "p_hat", p_hat)
            setattr(args, "loss_fn", loss_fn)
            setattr(args, "run_save_dir", run_save_dir)

            if args.wandb:
                logger = WandbLogger(project="symile", log_model="all",
                                    name=f"{loss_fn}_{datetime_now}",
                                    save_dir=run_save_dir)
            else:
                logger = False

            checkpoint_callback = ModelCheckpoint(dirpath=run_save_dir,
                                                  filename="{epoch}-{val_loss:.2f}",
                                                  mode="min",
                                                  monitor="val_loss")
            trainer = Trainer(
                callbacks=[checkpoint_callback],
                check_val_every_n_epoch=args.check_val_every_n_epoch,
                deterministic=True,
                enable_progress_bar=True,
                log_every_n_steps=1,
                logger=logger,
                max_epochs=args.epochs,
                num_sanity_val_steps=0
            )

            print(f"\n***** running {loss_fn} with p_hat = {p_hat}... *****\n")

            dm = BinaryDataModule(args)
            model = BinaryModule(**vars(args))

            trainer.fit(model, datamodule=dm)
            test_res = trainer.test(ckpt_path="best", datamodule=dm)[0]

            res["seed"].append(seed)
            res["p_hat"].append(p_hat)
            res["loss_fn"].append(loss_fn)
            res["acc"].append(test_res["mean_acc"])

            if args.wandb:
                logger.experiment.finish()

    return res


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'

    args = parse_args()

    datetime_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = args.save_dir / datetime_now
    os.makedirs(save_dir)
    print(f"\nResults will be saved in {save_dir}.\n")

    all_results = {"seed": [], "p_hat": [], "loss_fn": [], "acc": []}

    for seed in range(args.num_runs):
        res = main(seed, save_dir, args)

        for k in res:
            all_results[k] += res[k]

    acc_df = pd.DataFrame(all_results)
    acc_df.to_csv(save_dir / "acc.csv", index=False)

    if args.num_runs == 1:
        fig = px.line(acc_df, x="p_hat", y="acc", color="loss_fn")
        fig.write_image(save_dir / "acc.png")