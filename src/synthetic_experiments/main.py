"""
Experiment to demonstrate performance of SYMILE on synthetic datasets.
"""
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from sklearn.linear_model import LogisticRegression

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
            assert r_c is not None, "r_c must be defined for pretraining."
            loss = loss_fn(r_a, r_b, r_c, logit_scale, args.normalize)
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({"pretrain_loss": loss, "logit_scale": model.logit_scale.item()})
            optimizer.zero_grad()
            # TODO: how long to train for?

# def finetune(ft_loader, model, loss_fn, optimizer, args):
#     model.train()
#     for epoch in range(args.finetune_epochs):
#         for r_a, r_b, C in ft_loader:
#             C_pred = model(r_a, r_b)
#             assert C_pred.shape[0] == args.finetune_n, \
#                 "C_pred must have shape (n, 1)."
#             loss = loss_fn(C_pred, C.float().unsqueeze(1))
#             loss.backward()
#             optimizer.step()
#             if args.wandb:
#                 wandb.log({"finetune_loss": loss})
#             optimizer.zero_grad()
#             # TODO: how long to train for?

def finetune(ft_loader, model, loss_fn, optimizer, args):
    i = 0
    for r_a, r_b, C in ft_loader:
        lr.fit(torch.cat((r_a, r_b), dim=1), C)
        i += 1
    assert i == 1

# def test(test_loader, model, loss_fn, args):
#     # TODO: am I doing the loss calculation correctly?
#     model.eval()
#     n = len(test_loader.dataset)
#     loss = 0
#     with torch.no_grad():
#         for r_a, r_b, C in test_loader:
#             C_pred = model(r_a, r_b)
#             breakpoint()
#             loss += loss_fn(C_pred, C.float().unsqueeze(1)).item()
#     loss /= n
#     print("Test MSE: ", round(loss, 3))
#     if args.wandb:
#         wandb.log({"test_loss": loss})

def test(test_loader, model, loss_fn, args):
    i = 0
    for r_a, r_b, C in test_loader:
        mean_acc = lr.score(torch.cat((r_a, r_b), dim=1), C)
        print("Mean accuracy: ", mean_acc)
        if args.wandb:
            wandb.log({"mean_acc": mean_acc})
        i += 1
    assert i == 1

if __name__ == '__main__':
    # TODO: write tests for all experiment scripts
    # TODO: add comments
    args = parse_args()
    wandb_init(args)

    # pretraining
    pt_loader = load_data(args.d_v, args.pretrain_n, args.batch_sz, stage="pretrain")
    encoders = LinearEncoders(args.d_v, args.d_r)
    optimizer = torch.optim.AdamW(encoders.parameters(), lr=args.lr)
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    pretrain(pt_loader, encoders, loss_fn, optimizer, args)

    # finetuning
    ft_loader = load_data(args.d_v, args.finetune_n, args.batch_sz, stage="finetune",
                          model=encoders)
    # regression_model = LinearRegression(args.d_r)
    # finetune(ft_loader, regression_model, MSELoss(), optimizer, args)
    lr = LogisticRegression()
    finetune(ft_loader, lr, MSELoss(), optimizer, args)

    # testing
    test_loader = load_data(args.d_v, args.test_n, args.batch_sz, stage="test",
                            model=encoders)
    # test(test_loader, regression_model, MSELoss(reduction="sum"), args)
    test(test_loader, lr, None, args)