"""
Experiment to demonstrate performance of SYMILE on synthetic data.
"""
import torch
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from datasets import PretrainingDataset
from losses import pairwise_infonce, symile
from models import LinearEncoders
from params import parse_args, wandb_init


def load_data(d, n, batch_size):
    pt_dataset = PretrainingDataset(d, n)
    pt_loader = DataLoader(pt_dataset, batch_size=batch_size, shuffle=True)
    ft_loader = None
    return pt_loader, ft_loader

def pretrain(pt_loader, model, loss_fn, optimizer, args):
    model.train()
    for epoch in range(args.epochs):
        for v_a, v_b, v_c in pt_loader:
            r_a, r_b, r_c, logit_scale = model(v_a, v_b, v_c)

            loss = loss_fn(r_a, r_b, r_c, logit_scale, args.normalize)
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({"loss": loss, "logit_scale": model.logit_scale.item()})
            optimizer.zero_grad()

if __name__ == '__main__':
    # TODO: write tests for all experiment scripts
    args = parse_args()
    wandb_init(args)

    pt_loader, ft_loader = load_data(args.d_v, args.n, args.batch_sz)
    model = LinearEncoders(args.d_v, args.d_r)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    pretrain(pt_loader, model, loss_fn, optimizer, args)