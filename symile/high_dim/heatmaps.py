import os
from pathlib import Path

from lightning.pytorch import seed_everything

from args import parse_args_main
from datasets import HighDimDataModule
from models import SSLModel


def data_to_device(batch, device):
    batch["image"] = batch["image"].to(device)
    batch["audio"] = batch["audio"].to(device)
    for k in batch["text"]:
        batch["text"][k] = batch["text"][k].to(device)
    return batch


if __name__ == '__main__':
    args = parse_args_main()

    seed_everything(args.seed, workers=True)

    save_dir = Path(os.path.dirname(args.load_from_ckpt) + "/heatmaps")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("\nSaving to: ", save_dir)

    dm = HighDimDataModule(args)
    dm.setup(stage='test')

    model = SSLModel.load_from_checkpoint(args.load_from_ckpt)
    model.eval()

    batch = next(iter(dm.test_dataloader()))
    batch = data_to_device(batch, model.device)

    model.test_step(batch, 0, save_test_heatmaps=True, save_dir=save_dir)