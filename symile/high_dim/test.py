from datetime import datetime
import os
import random
import time

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from args import parse_args_test
from datasets import HighDimDataset
from models import SSLModel


def get_dataloader(args):
    ds_test = HighDimDataset(args, "test")

    num_workers = len(os.sched_getaffinity(0))

    if args.bootstrap:
        sampler = WeightedRandomSampler(torch.ones(len(ds_test)),
                                        num_samples=len(ds_test),
                                        replacement=True)
        return DataLoader(ds_test, batch_size=args.batch_sz_test,
                          sampler=sampler, num_workers=num_workers, drop_last=False)
    else:
        return DataLoader(ds_test, batch_size=self.args.batch_sz_test,
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

    # if args.use_seed:
    #     seed_everything(args.seed, workers=True)
    test_acc = test(args, trainer)[0]["test_accuracy"]

    if args.bootstrap:
        accuracies = []

        for i in range(args.bootstrap_n):
            print(f"\nbootstrap {i+1}/{args.bootstrap_n}")

            # if args.use_seed:
            #     seed_everything(args.seed + i + 1, workers=True)

            acc = test(args, trainer)[0]["test_accuracy"]
            accuracies.append(acc)

        ci_low = np.percentile(accuracies, 2.5)
        ci_high = np.percentile(accuracies, 97.5)

        print("\nsaving results to ", save_dir / "results.txt")
        print(f"\ntest accuracy: {test_acc:.4f}")
        print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
        with open(save_dir / "results.txt", "w") as f:
            f.write(f"test accuracy: {test_acc:.4f}\n")
            f.write(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
    else:
        print("\nsaving results to ", save_dir / "results.txt")
        print(f"\ntest accuracy: {test_acc:.4f}")
        with open(save_dir / "results.txt", "w") as f:
            f.write(f"test accuracy: {test_acc:.4f}")

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")