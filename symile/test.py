from datetime import datetime
import importlib
import json
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader

from args import parse_args_test
import datasets


def get_dataloader(args):
    """
    Loads and returns a dataloader instance based on the experiment name.
    """
    if args.experiment == "symile_m3":
        ds_test = datasets.SymileM3Dataset(args, "test")
    elif args.experiment == "cxr_prediction":
        dm = datasets.CXRPredictionDataModule(args)
        dm.setup(stage="test")
        ds_test = dm.ds_test
    else:
        raise ValueError("Unsupported experiment name specified.")

    num_workers = len(os.sched_getaffinity(0))

    return DataLoader(ds_test, batch_size=args.batch_sz_test,
                      shuffle=False, num_workers=num_workers, drop_last=False)


def load_model_from_ckpt(args):
    """
    Loads and returns a model instance from a checkpoint file based on experiment name.
    """
    if args.experiment == "symile_m3":
        module = importlib.import_module("models.symile_m3_model")
        ModelClass = getattr(module, "SymileM3Model")
    elif args.experiment == "cxr_prediction":
        module = importlib.import_module("models.cxr_prediction_model")
        ModelClass = getattr(module, "CXRPredictionModel")
    else:
        raise ValueError("Unsupported experiment name specified.")

    return ModelClass.load_from_checkpoint(args.load_from_ckpt)


def test(args, trainer):
    print("Loading checkpoint from ", args.load_from_ckpt)
    model = load_model_from_ckpt(args)

    # override model args
    model.args.data_dir = args.data_dir
    model.args.save_dir = args.save_dir
    model.args.save_representations = args.save_representations

    # set dl as an attribute of the model
    dl = get_dataloader(args)
    model.test_dataloader = dl

    model.eval()
    return trainer.test(model, dataloaders=dl)


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

    args = parse_args_test()

    randint = random.randint(0, 9999) # reduces chance of directory name collision when scripts are run in parallel
    save_dir = args.save_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{randint:04d}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    setattr(args, "save_dir", save_dir)
    print("\nSaving to: ", save_dir)

    trainer = Trainer(
        deterministic=args.use_seed,
        enable_progress_bar=True,
        logger=False
    )

    if args.use_seed:
        seed_everything(args.seed, workers=True)

    metrics = test(args, trainer)[0]
    metrics["description"] = args.description

    save_pt = save_dir / "results.json"
    print("\nsaving results to ", save_pt)

    with open(save_pt, "w") as f:
        json.dump(metrics, f, indent=4)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")