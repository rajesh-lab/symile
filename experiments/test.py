import csv
from datetime import datetime
import importlib
import json
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from torch.utils.data import DataLoader, RandomSampler

from args import parse_args_test
import datasets


def get_dataloader(args):
    """
    Loads and returns a dataloader instance based on the experiment name.
    """
    num_workers = len(os.sched_getaffinity(0))

    if args.experiment == "symile_m3":
        ds_test = datasets.SymileM3Dataset(args, "test")
    elif args.experiment == "symile_mimic":
        dm = datasets.SymileMIMICDataModule(args)
        dm.setup(stage="test")
        ds_test = dm.ds_test
    else:
        raise ValueError("Unsupported experiment name specified.")

    if args.bootstrap:
        sampler = RandomSampler(ds_test, replacement=True, num_samples=len(ds_test))
        dl = DataLoader(ds_test, sampler=sampler, batch_size=args.batch_sz_test,
                        shuffle=False, num_workers=num_workers, drop_last=False)
    else:
        dl = DataLoader(ds_test, batch_size=args.batch_sz_test,
                        shuffle=False, num_workers=num_workers, drop_last=False)

    return dl


def load_model_from_ckpt(args):
    """
    Loads and returns a model instance from a checkpoint file based on experiment name.
    """
    if args.experiment == "symile_m3":
        module = importlib.import_module("models.symile_m3_model")
        ModelClass = getattr(module, "SymileM3Model")
    elif args.experiment == "symile_mimic":
        module = importlib.import_module("models.symile_mimic_model")
        ModelClass = getattr(module, "SymileMIMICModel")
    else:
        raise ValueError("Unsupported experiment name specified.")

    return ModelClass.load_from_checkpoint(args.ckpt_path,
                                           batch_sz_test=args.batch_sz_test,
                                           data_dir=args.data_dir,
                                           save_dir=args.save_dir,
                                           bootstrap=args.bootstrap)


def test(args, trainer):
    print("\nLoading checkpoint from ", args.ckpt_path)
    model = load_model_from_ckpt(args)

    # set dl as an attribute of the model
    dl = get_dataloader(args)
    model.test_dataloader = dl

    model.eval()
    return trainer.test(model, dataloaders=dl)


if __name__ == '__main__':
    start = time.time()

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

    if args.bootstrap:
        bootstrap_metrics = []

        for i in range(args.bootstrap_n):
            print(f"\nRunning bootstrap iteration {i+1}/{args.bootstrap_n}...")

            metrics = test(args, trainer)[0]

            bootstrap_metrics.append(metrics)

        headers = list(bootstrap_metrics[0].keys())
        save_pt = save_dir / "bootstrap_metrics.csv"
        with open(save_pt, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(bootstrap_metrics)
    else:
        metrics = test(args, trainer)[0]
        metrics["description"] = args.description

        save_pt = save_dir / "results.json"
        print("\nsaving results to ", save_pt)

        with open(save_pt, "w") as f:
            json.dump(metrics, f, indent=4)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")