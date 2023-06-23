"""
Experiment to demonstrate performance of SYMILE on synthetic data.
"""
from sklearn.linear_model import LinearRegression
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


def load_data(args):
    pt_dataset = PretrainingDataset(args.d_v, args.pretrain_n)
    pt_loader = DataLoader(pt_dataset, batch_size=args.batch_sz, shuffle=True)

    ft_dataset = PretrainingDataset(args.d_v, args.finetune_n)
    ft_loader = DataLoader(ft_dataset, batch_size=args.batch_sz, shuffle=True)
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

def finetune(ft_loader, model, args):
    model.eval()
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    with torch.no_grad():
        i = 0
        for v_a, v_b, v_c in ft_loader:
            r_a, r_b, r_c, logit_scale = model(v_a, v_b, v_c)
            if i == 0:
                X_test = torch.cat((r_a,r_b), dim=1)
                y_test = r_c
            else:
                if X_train is None:
                    X_train = torch.cat((r_a,r_b), dim=1)
                else:
                    X_train = torch.cat((X_train, torch.cat((r_a,r_b), dim=1)), dim=0)
                if y_train is None:
                    y_train = r_c
                else:
                    y_train = torch.cat((y_train, r_c), dim=0)
            i += 1
    reg = LinearRegression().fit(X_train, y_train)
    print("R^2 score: {}".format(reg.score(X_test, y_test)))


if __name__ == '__main__':
    # TODO: write tests for all experiment scripts
    args = parse_args()
    wandb_init(args)

    pt_loader, ft_loader = load_data(args)
    model = LinearEncoders(args.d_v, args.d_r)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    pretrain(pt_loader, model, loss_fn, optimizer, args)

    reg = finetune(ft_loader, model, args)
