import os
import re

import pandas as pd
import yaml

from args import parse_args_collect_tuning_results


def main(args, tuning_data):
    data = []

    # regular expression to parse the ckpt filename
    ckpt_pattern = re.compile(r"^epoch=(\d+)-val_loss=([\d.]+)\.ckpt$")

    for loss_fn, runs in tuning_data.items():
        for run_name, run_details in runs.items():
            ckpt_dir = run_details["ckpt_dir"]

            try: # get all ckpt files for current run
                for ckpt_filename in os.listdir(ckpt_dir):
                    if ckpt_filename.endswith(".ckpt"):
                        # Parse the filename to get epoch and validation loss
                        match = ckpt_pattern.search(ckpt_filename)
                        if match:
                            epoch, val_loss = match.groups()

                            ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

                            data.append({
                                "run_name": run_name,
                                "loss_fn": loss_fn,
                                "ckpt_path": ckpt_path,
                                "epoch_number": int(epoch),
                                "val_loss": float(val_loss),
                                "weight_decay": run_details["wd"],
                                "learning_rate": run_details["lr"],
                                "wandb_path": run_details["wandb"],
                                "best_ckpt": run_details["best_ckpt"]
                            })
            except FileNotFoundError:
                print(f"Directory not found: {ckpt_dir}")

    return pd.DataFrame(data)

if __name__ == '__main__':
    args = parse_args_collect_tuning_results()

    # Read YAML file
    with open(args.results_pt, "r") as file:
        tuning_data = yaml.safe_load(file)

    df = main(args, tuning_data)

    df.to_csv(args.save_dir / "tuning_runs.csv", index=False)