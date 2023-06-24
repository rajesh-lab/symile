"""
Experiment to demonstrate performance of SYMILE on synthetic data.
"""
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from datasets import FinetuningDataset, PretrainingDataset
from losses import pairwise_infonce, symile
from models import LinearEncoders, LinearRegression
from params import parse_args, wandb_init


def load_data(d, n, batch_sz, stage="pretrain", model=None):
    if stage == "pretrain":
        dataset = PretrainingDataset(d, n)
    elif stage == "finetune" or stage == "test":
        dataset = FinetuningDataset(d, n, model)
    return DataLoader(dataset, batch_size=batch_sz, shuffle=True)

def pretrain(pt_loader, model, loss_fn, optimizer, args):
    model.train()
    for epoch in range(args.pretrain_epochs):
        for v_a, v_b, v_c in pt_loader:
            r_a, r_b, r_c, logit_scale = model(v_a, v_b, v_c)

            loss = loss_fn(r_a, r_b, r_c, logit_scale, args.normalize)
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({"pretrain_loss": loss, "logit_scale": model.logit_scale.item()})
            optimizer.zero_grad()
            # TODO: how long to train for?

def finetune(ft_loader, model, loss_fn, optimizer, args):
    model.train()
    for epoch in range(args.finetune_epochs):
        for r_a, r_b, C in ft_loader:
            C_pred = model(r_a, r_b)

            loss = loss_fn(C_pred, C.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({"finetune_loss": loss})
            optimizer.zero_grad()
            # TODO: how long to train for?

def test(test_loader, model, loss_fn, args):
    model.eval()
    n = len(test_loader.dataset)
    loss = 0
    with torch.no_grad():
        for r_a, r_b, C in test_loader:
            C_pred = model(r_a, r_b)
            loss += loss_fn(C_pred, C.float().unsqueeze(1)).item()
    loss /= n
    print("Test MSE: ", round(loss, 3))

if __name__ == '__main__':
    # TODO: write tests for all experiment scripts
    args = parse_args()
    wandb_init(args)

    # pretraining
    pt_loader = load_data(args.d_v, args.pretrain_n, args.batch_sz, stage="pretrain")
    encoders = LinearEncoders(args.d_v, args.d_r)
    optimizer = torch.optim.AdamW(encoders.parameters(), lr=args.lr)
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    pretrain(pt_loader, encoders, loss_fn, optimizer, args)

    # finetuning
    ft_loader = load_data(args.d_v, args.finetune_n, args.batch_sz,
                          stage="finetune", model=encoders)
    regression_model = LinearRegression(args.d_r)
    finetune(ft_loader, regression_model, MSELoss(), optimizer, args)

    # testing
    test_loader = load_data(args.d_v, args.test_n, args.batch_sz,
                            stage="test", model=encoders)
    test(test_loader, regression_model, MSELoss(reduction="sum"), args)