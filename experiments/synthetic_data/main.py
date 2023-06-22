"""
Experiment to demonstrate performance of SYMILE on synthetic data.
"""
import torch
from torch.utils.data import DataLoader

from datasets import PretrainingDataset
from losses import pairwise_infonce, symile
from models import LinearEncoders


def load_data(d, n, p, batch_size):
    pt_dataset = PretrainingDataset(d, n, p)
    pt_loader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=True)
    ft_loader = None
    return pt_loader, ft_loader

def pretrain(pt_loader, model, loss_fn, optimizer, epochs, normalize):
    model.train()
    for epoch in range(epochs):
        for v_a, v_b, v_c in pt_loader:
            r_a, r_b, r_c, logit_scale = model(v_a, v_b, v_c)

            loss = loss_fn(r_a, r_b, r_c, logit_scale, normalize)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

if __name__ == '__main__':
    # TODO: write tests for scripts
    # TODO: move these arguments into a json file or argparse
    d_v = 5
    d_r = 2
    n = 10
    p = 0.5
    batch_size = 2
    lr = 0.1
    loss_fn = "symile"
    epochs = 2
    normalize = True

    pt_loader, ft_loader = load_data(d_v, n, p, batch_size)
    model = LinearEncoders(d_v, d_r)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = symile if loss_fn == "symile" else pairwise_infonce
    pretrain(pt_loader, model, loss_fn, optimizer, epochs, normalize)

    print("Done!")