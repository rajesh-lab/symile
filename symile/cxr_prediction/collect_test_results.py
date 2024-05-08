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


def main(args, test_data):
    data = []

    for loss_fn, paths in test_data.items():
        for seed, path in paths.items():
            run_name = os.path.basename(path)
            results_path = os.path.join(path, "results.json")
            with open(results_path, "r") as f:
                results = json.load(f)

            for key, value in results.items():
                if key == 'description':
                    continue

                parts = key.split('_at_')
                if len(parts) == 2:
                    metric_name, k = parts
                    # Remove 'test_' prefix if present
                    metric_name = metric_name[5:] if metric_name.startswith('test_') else metric_name
                    # Append structured data
                    data.append({
                        'loss_fn': loss_fn,
                        'seed': seed,
                        'run_name': run_name,
                        'metric': metric_name,
                        'value': value,
                        'k': int(k)  # Convert k to integer
                    })

    return pd.DataFrame(data)

if __name__ == '__main__':
    args = parse_args_collect_tuning_results()

    # Read YAML file
    with open(args.results_pt, "r") as file:
        test_data = yaml.safe_load(file)

    df = main(args, test_data)

    df.to_csv(args.save_pt, index=False)