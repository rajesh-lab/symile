import glob
import json
import os
import re

import pandas as pd
import yaml

from args import parse_args_collect_tuning_results


def find_ckpt_file(run_path, epoch):
    # all '.ckpt' files in the directory
    ckpt_files = glob.glob(os.path.join(run_path, "*.ckpt"))

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {run_path}")

    sample_ckpt = os.path.basename(ckpt_files[0])

    if "val_acc" in sample_ckpt:
        pattern = re.compile(rf"epoch={epoch}-val_loss=\d+\.\d+-val_acc=\d+\.\d+\.ckpt$")
    elif "val_loss" in sample_ckpt:
        pattern = re.compile(rf"epoch={epoch}-val_loss=\d+\.\d+\.ckpt$")
    else:
        pattern = re.compile(rf"epoch={epoch}-.*\.ckpt$")

    matched_files = [f for f in ckpt_files if pattern.search(os.path.basename(f))]

    assert len(matched_files) == 1, \
        f"Expected exactly one matching checkpoint, found {len(matched_files)}."

    return matched_files[0] if matched_files else None


def assert_matching_val_loss(val_loss, ckpt_pt, experiment):
     # extract val_loss from ckpt_pt
    match = re.search(r'val_loss=([0-9]+\.[0-9]+)', ckpt_pt)
    if match:
        file_val_loss = float(match.group(1))
    else:
        raise ValueError("No valid val_loss found in the checkpoint path.")

    if experiment == "symile_m3":
        decimal_places = len(match.group(1).split('.')[1])
    elif experiment == "symile_mimic":
        decimal_places = len(ckpt_pt.split('=')[-1].split('.')[1].rstrip('.ckpt'))

    # assert that the rounded values match
    assert file_val_loss == round(val_loss, decimal_places), \
        f"Validation loss from file ({round(file_val_loss, decimal_places)}) does not \
          match the loss from metrics ({round(val_loss, decimal_places)})."


def main_symile_m3(tuning_data, experiment):
    data = []

    # regular expression to parse the ckpt filename
    ckpt_pattern = re.compile(r"^epoch=(\d+)-val_loss=([\d.]+)\.ckpt$")

    for loss_fn, paths in tuning_data.items():
        for path in paths:
            run_name = os.path.basename(path)
            run_info_path = os.path.join(path, "run_info.json")
            with open(run_info_path, "r") as f:
                run_info = json.load(f)

            for val_metric in run_info["validation_metrics"]:
                missingness = run_info['args']['missingness']
                ckpt_pt = find_ckpt_file(path, val_metric["epoch"])

                if not missingness:
                    assert loss_fn == run_info['args']['loss_fn'], "Mismatch in loss function."
                assert_matching_val_loss(val_metric['val_loss'], ckpt_pt, experiment)

                # Prepare a dictionary for each row
                val_accuracy = val_metric['val_acc'] if missingness else val_metric['val_accuracy']
                row_data = {
                    'run_name': run_name,
                    'loss_fn': loss_fn,
                    'ckpt_pt': ckpt_pt,
                    'epoch': val_metric['epoch'],
                    'val_loss': val_metric['val_loss'],
                    'val_accuracy': val_accuracy,
                    'weight_decay': run_info['args']['weight_decay'],
                    'learning_rate': run_info['args']['lr'],
                    'wandb_path': run_info['wandb']
                }

                data.append(row_data)

    return pd.DataFrame(data)


def main_symile_mimic(tuning_data, experiment):
    data = []

    for loss_fn, paths in tuning_data.items():
        for path in paths:
            run_name = os.path.basename(path)
            run_info_path = os.path.join(path, "run_info.json")
            with open(run_info_path, "r") as f:
                run_info = json.load(f)

            for val_metric in run_info["validation_metrics"]:
                ckpt_pt = find_ckpt_file(path, val_metric['epoch'], experiment)

                assert loss_fn == run_info['args']['loss_fn'], "Mismatch in loss function."
                assert_matching_val_loss(val_metric['val_loss'], ckpt_pt, experiment)

                row_data = {
                    'run_name': run_name,
                    'loss_fn': loss_fn,
                    'ckpt_pt': ckpt_pt,
                    'pretrained': run_info['args']['pretrained'],
                    'val_loss': val_metric['val_loss'],
                    'val_accuracy': val_metric['val_acc'],
                    'weight_decay': run_info['args']['weight_decay'],
                    'learning_rate': run_info['args']['lr'],
                    'wandb_path': run_info['wandb'],
                    'seed': run_info['args']['seed']
                }

                data.append(row_data)

    return pd.DataFrame(data)


if __name__ == '__main__':
    args = parse_args_collect_tuning_results()

    # Read YAML file
    with open(args.results_pt, "r") as file:
        tuning_data = yaml.safe_load(file)

    if args.experiment == "symile_m3":
        df = main_symile_m3(tuning_data, args.experiment)
    elif args.experiment == "symile_mimic":
        df = main_symile_mimic(tuning_data, args.experiment)
    else:
        raise ValueError("Invalid experiment.")

    df.to_csv(args.save_pt, index=False)