import time

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import umap

from args import parse_args_create_umap_reps


def apply_umap_and_save(tensor, save_pt, args):
    tensor = tensor.numpy()

    print("    standardizing data...\n")
    scaler = StandardScaler()
    tensor_scaled = scaler.fit_transform(tensor)

    print("    initializing UMAP...\n")
    reducer = umap.UMAP(n_neighbors=15,
                        n_components=args.n_components,
                        metric=args.metric,
                        min_dist=0.1)
    embedding = reducer.fit_transform(tensor_scaled)

    print("    saving embedding...\n")
    np.save(save_pt, embedding)


if __name__ == '__main__':
    start = time.time()

    args = parse_args_create_umap_reps()

    if not args.save_dir.exists():
        args.save_dir.mkdir(parents=True)

    for mode in ["r_a", "r_i", "r_t"]:
        print(f"Processing {mode}...\n")
        rep = torch.load(args.rep_dir / f"{mode}_test.pt")[:args.n_datapoints] # (batch_sz, n_datapoints)
        apply_umap_and_save(rep, args.save_dir / f"{mode}_{args.n_components}d_{args.metric}", args)

    end = time.time()
    total_time = (end - start)/60
    print(f"Script took {total_time:.4f} minutes")
