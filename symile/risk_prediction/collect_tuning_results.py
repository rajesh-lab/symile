import json
import os
import re
import glob

import pandas as pd
import yaml

from args import parse_args_collect_tuning_results


def find_ckpt_file(run_path, epoch):
    # match 'epoch={epoch}-' and ends with '.ckpt'
    pattern = re.compile(rf"epoch={epoch}-.*\.ckpt$")

    # all '.ckpt' files in the directory
    ckpt_files = glob.glob(os.path.join(run_path, "*.ckpt"))

    matched_files = [f for f in ckpt_files if pattern.search(os.path.basename(f))]

    assert len(matched_files) == 1, \
        f"Expected exactly one matching checkpoint, found {len(matched_files)}."

    return matched_files[0] if matched_files else None


def assert_matching_val_loss(val_loss, ckpt_pt):
     # extract val_loss from ckpt_pt
    match = re.search(r'val_loss=([0-9]+\.[0-9]+)', ckpt_pt)
    if match:
        file_val_loss = float(match.group(1))
    else:
        raise ValueError("No valid val_loss found in the checkpoint path.")

    decimal_places = len(ckpt_pt.split('=')[-1].split('.')[1].rstrip('.ckpt'))

    # assert that the rounded values match
    assert file_val_loss == round(val_loss, decimal_places), \
        f"Validation loss from file ({round(file_val_loss, decimal_places)}) does not \
          match the loss from metrics ({round(val_loss, decimal_places)})."


def main(args, tuning_data):
    data = []

    for loss_fn, paths in tuning_data.items():
        for path in paths:
            run_name = os.path.basename(path)
            run_info_path = os.path.join(path, "run_info.json")
            with open(run_info_path, "r") as f:
                run_info = json.load(f)

            assert len(run_info["validation_metrics"]) == 50, \
                f"Expected 50 validation metrics, found {len(run_info['validation_metrics'])}."

            for val_metric in run_info["validation_metrics"]:
                ckpt_pt = find_ckpt_file(path, val_metric['epoch'])

                assert_matching_val_loss(val_metric['val_loss'], ckpt_pt)

                # Prepare a dictionary for each row
                row_data = {
                    'run_name': run_name,
                    'loss_fn': loss_fn,
                    'ckpt_pt': ckpt_pt,
                    'epoch': val_metric['epoch'],
                    'val_loss': val_metric['val_loss'],
                    'val_acc_at_1': val_metric['val_acc_at_1'],
                    'val_acc_at_5': val_metric['val_acc_at_5'],
                    'val_acc_at_10': val_metric['val_acc_at_10'],
                    'weight_decay': run_info['args']['weight_decay'],
                    'learning_rate': run_info['args']['lr'],
                    'wandb_path': run_info['wandb']
                }

                data.append(row_data)

    return pd.DataFrame(data)

if __name__ == '__main__':
    args = parse_args_collect_tuning_results()

    # Read YAML file
    with open(args.results_pt, "r") as file:
        tuning_data = yaml.safe_load(file)

    df = main(args, tuning_data)

    df.to_csv(args.save_dir / "tuning_runs.csv", index=False)