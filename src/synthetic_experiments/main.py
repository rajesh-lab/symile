"""
Experiment to demonstrate performance of SYMILE on synthetic datasets.
"""
from sklearn.neural_network import MLPClassifier
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from datasets import FinetuningDataset, PretrainingDataset
from losses import pairwise_infonce, symile
from models import LinearEncoders
from params import parse_args, wandb_init


def load_data(d, n, batch_sz, stage="pretrain", model=None):
    if stage == "pretrain":
        dataset = PretrainingDataset(d, n)
    elif stage == "finetune" or stage == "test":
        dataset = FinetuningDataset(d, n, model)
    return DataLoader(dataset, batch_size=batch_sz, shuffle=True)

def pretrain(pt_loader, pt_val_loader, model, loss_fn, optimizer, args):
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

def finetune(ft_loader, model):
    for r_a, r_b, C_bin in ft_loader:
        model.fit(torch.cat((r_a, r_b), dim=1), C_bin)

def test(test_loader, model):
    for r_a, r_b, C_bin in test_loader:
        mean_acc = model.score(torch.cat((r_a, r_b), dim=1), C_bin)
        print("Mean accuracy: ", mean_acc)
        if args.wandb:
            wandb.log({"mean_acc": mean_acc})

if __name__ == '__main__':
    args = parse_args()
    wandb_init(args)

    # set batch sizes
    batch_sz_pt = args.pretrain_n if args.use_full_dataset else args.batch_sz_pt

    # pretraining
    pt_loader = load_data(args.d_v, args.pretrain_n, batch_sz_pt, stage="pretrain")
    pt_val_loader = load_data(args.d_v, args.pretrain_val_n, args.pretrain_val_n, stage="pretrain")
    encoders = LinearEncoders(args.d_v, args.d_r)
    optimizer = torch.optim.AdamW(encoders.parameters(), lr=args.lr)
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    pretrain(pt_loader, pt_val_loader, encoders, loss_fn, optimizer, args)

    # finetuning
    ft_loader = load_data(args.d_v, args.finetune_n, args.finetune_n, stage="finetune",
                          model=encoders)
    clf = MLPClassifier([100,100])
    finetune(ft_loader, clf)

    # testing
    test_loader = load_data(args.d_v, args.test_n, args.test_n, stage="test",
                            model=encoders)
    test(test_loader, clf)