import os

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch

from args import parse_args_main
from datasets import SymileDataModule
from models import SymileModel


def main(args):
    checkpoint_callback = ModelCheckpoint(filename="{epoch}-{val_loss:.2f}",
                                          mode="min",
                                          monitor="val_loss")
    early_stopping_callback = EarlyStopping(monitor="val_loss",
                                            mode="min",
                                            patience=args.early_stopping_patience)
    logger = WandbLogger(project="symile", log_model="all") if args.wandb else None
    # `ddp_find_unused_parameters_true` instead of `ddp` because error is thrown
    # when not all indices of nn.Embedding are used in a minibatch
    profiler = None if args.profiler == "none" else args.profiler
    strategy="ddp_find_unused_parameters_true" if torch.cuda.device_count() > 1 else "auto"

    trainer = Trainer(
        callbacks=[checkpoint_callback, early_stopping_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        deterministic=args.use_seed,
        enable_progress_bar=True,
        logger=logger,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        profiler=profiler,
        strategy=strategy
    )

    dm = SymileDataModule(args)
    dm.setup(stage="fit")
    args.feat_token_id = dm.feat_token_id

    symile_model = SymileModel(**vars(args))
    if args.wandb:
        logger.watch(symile_model)

    # PRETRAIN
    trainer.fit(symile_model, datamodule=dm)

    # # EVALUATE
    # PASS IN ALL EVALUATIONS YOU WANT TO DO AS LIST BECAUSE YOU DON'T WANT TO TRAIN AGAIN
    # # automatically loads the best weights for you?? if checkpointing was enabled during fitting?
    # # dm.setup(stage="test") # do I need this?
    # trainer.test(model=symile_model, datamodule=dm)


    # # maybe move this into test loop?? if you can load different test data???
    # if args.evaluation == "zeroshot_clf":
    #     print("\n\n...evaluation: zero-shot classification...\n")
    #     test_zeroshot_clf(args, symile_model)
    # # elif args.evaluation == "support_clf":
    # #     print("\n\n\n...evaluation: in support classification...\n")
    # #     test_support_clf(args, encoders)


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'

    args = parse_args_main()

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    main(args)