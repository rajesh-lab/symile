from datetime import datetime
import json
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from args import parse_args_test
from datasets import CXRPredictionDataModule
from models import SSLModel


def get_dataloader(args):
    dm = CXRPredictionDataModule(args)
    dm.setup(stage="test")
    ds_test = dm.ds_test

    num_workers = len(os.sched_getaffinity(0))

    if args.bootstrap:
        sampler = WeightedRandomSampler(torch.ones(len(ds_test)),
                                        num_samples=len(ds_test),
                                        replacement=True)
        return DataLoader(ds_test, batch_size=args.batch_sz_test,
                          sampler=sampler, num_workers=num_workers, drop_last=False)
    else:
        return DataLoader(ds_test, batch_size=args.batch_sz_test,
                          shuffle=False, num_workers=num_workers, drop_last=False)

def test(args, trainer):
    print("Loading checkpoint from ", args.load_from_ckpt)
    model = SSLModel.load_from_checkpoint(args.load_from_ckpt)

    # override model args
    model.args.data_dir = args.data_dir
    model.args.save_dir = args.save_dir

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

    if args.bootstrap:
        # NOTE: doesn't make sense to bootstrap here because seed isn't doing anything!!!
        metric_data = {key: [] for key in metrics.keys()}

        for i in range(args.bootstrap_n):
            print(f"\nbootstrap {i+1}/{args.bootstrap_n}")

            seed_everything(args.seed + i + 1, workers=True)

            bs_results = test(args, trainer)[0]

            for key in metrics.keys():
                metric_data[key].append(bs_results[key])

        print("\nsaving results to ", save_dir / "results.txt")

        with open(save_dir / "results.txt", "w") as f:
            for key in metrics.keys():
                ci_low = np.percentile(metric_data[key], 2.5)
                ci_high = np.percentile(metric_data[key], 97.5)
                mean_acc = np.mean(metric_data[key])

                f.write(f"{key}:\n")
                f.write(f"Mean: {mean_acc}\n")
                f.write(f"95% CI: [{ci_low}, {ci_high}]\n")
    else:
        save_pt = save_dir / "results.json"
        print("\nsaving results to ", save_pt)

        metrics["description"] = args.description

        with open(save_pt, "w") as f:
            json.dump(metrics, f, indent=4)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")