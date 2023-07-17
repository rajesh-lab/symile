"""
Experiment to demonstrate performance of SYMILE on synthetic datasets.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
import torch
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from datasets import FinetuningDataset, PretrainingDataset
from losses import pairwise_infonce, symile
from models import LinearEncoders
from params import parse_args, wandb_init


def load_data(i, args, stage="pretrain", model=None):
    if stage == "pretrain":
        dataset = PretrainingDataset(i, args.d_v, args.pretrain_n, args.eps_param, args.example_mod)
        batch_sz = args.pretrain_n if args.use_full_dataset else args.batch_sz_pt
        dl = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
    elif stage == "pretrain_val":
        dataset = PretrainingDataset(i, args.d_v, args.pretrain_val_n, args.eps_param, args.example_mod)
        dl = DataLoader(dataset, batch_size=args.pretrain_val_n, shuffle=True)
    elif stage == "finetune":
        dataset = FinetuningDataset(i, args.d_v, args.finetune_n, args.eps_param, args.example_mod, model)
        dl = DataLoader(dataset, batch_size=args.finetune_n, shuffle=True)
    elif stage == "test":
        dataset = FinetuningDataset(i, args.d_v, args.test_n, args.eps_param, args.example_mod, model)
        dl = DataLoader(dataset, batch_size=args.test_n, shuffle=True)
    return dl

def pretrain(pt_loader, pt_val_loader, model, loss_fn, optimizer, args, i):
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(args.pt_epochs):
        for v_a, v_b, v_c in pt_loader:
            r_a, r_b, r_c, logit_scale = model(v_a, v_b, v_c)
            assert r_c is not None, "r_c must be defined for pretraining."
            loss = loss_fn(r_a, r_b, r_c, logit_scale, args.normalize)
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({"pretrain_loss": loss, "logit_scale": model.logit_scale.item()})
            optimizer.zero_grad()

            if epoch % 20 == 0:
                print("epoch: ", epoch)
            if epoch % args.pt_val_epochs == 0:
                model.eval()
                with torch.no_grad():
                    for v_a, v_b, v_c in pt_val_loader:
                        r_a, r_b, r_c, logit_scale = model(v_a, v_b, v_c)
                        val_loss = loss_fn(r_a, r_b, r_c, logit_scale, args.normalize)
                        if args.wandb:
                            wandb.log({"pretrain_val_loss": val_loss})
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                if patience_counter >= args.early_stopping_patience_pt:
                    break
                else:
                    model.train()

def finetune(ft_loader, model, args):
    for A, B, C, r_a, r_b, r_c, C_bin in ft_loader:
        if args.pred_from_reps:
            X = torch.cat((r_a, r_b), dim=1)
        else:
            X = torch.stack((A, B), dim=1)
        if args.example_mod:
            model.fit(X, C_bin)
            print("finetuning mean acc: ", model.score(X, C_bin))
        else:
            model.fit(X, C)
            print("finetuning mean acc: ", model.score(X, C))

def test(test_loader, model, args):
    for A, B, C, r_a, r_b, r_c, C_bin in test_loader:
        if args.pred_from_reps:
            X = torch.cat((r_a, r_b), dim=1)
        else:
            X = torch.stack((A, B), dim=1)
        if args.example_mod:
            mean_acc = model.score(X, C_bin)
        else:
            mean_acc = model.score(X, C)
        print("Mean accuracy: ", mean_acc)
        if args.wandb:
            wandb.log({"mean_acc": mean_acc})

if __name__ == '__main__':
    args = parse_args()
    wandb_init(args)

    # sample i for data generation
    i = np.random.randint(0, args.d_v)

    # pretraining
    encoders = LinearEncoders(args.d_v, args.d_r)
    if args.pred_from_reps:
        print("\n\n\n...pretraining...\n")
        pt_loader = load_data(i, args, stage="pretrain")
        pt_val_loader = load_data(i, args, stage="pretrain_val")
        optimizer = torch.optim.AdamW(encoders.parameters(), lr=args.lr)
        loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
        pretrain(pt_loader, pt_val_loader, encoders, loss_fn, optimizer, args, i)

    # finetuning
    print("\n\n\n...finetuning...\n")
    ft_loader = load_data(i, args, stage="finetune", model=encoders)
    clf = MLPClassifier([100])
    finetune(ft_loader, clf, args)

    # testing
    print("\n\n\n...evaluating...\n")
    test_loader = load_data(i, args, stage="test", model=encoders)
    test(test_loader, clf, args)