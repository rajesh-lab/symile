"""
Experiment to demonstrate performance of SYMILE on synthetic datasets.
"""
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from datasets import PretrainingDataset, SumTestDataset, \
                     SupportTestDataset, ZeroshotTestDataset
from losses import pairwise_infonce, symile
from models import LinearEncoders
from params import parse_args, wandb_init


def load_data(args, stage="pretrain", model=None):
    if stage == "pretrain":
        dataset = PretrainingDataset(args.d_v, args.pretrain_n)
        batch_sz = args.pretrain_n if args.use_full_dataset else args.batch_sz_pt
        dl = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
    elif stage == "pretrain_val":
        dataset = PretrainingDataset(args.d_v, args.pretrain_val_n)
        dl = DataLoader(dataset, batch_size=args.pretrain_val_n, shuffle=True)
    elif stage == "finetune":
        dataset = FinetuningDataset(i, args.d_v, args.finetune_n, args.eps_param, args.example_mod, model)
        dl = DataLoader(dataset, batch_size=args.finetune_n, shuffle=True)
    elif stage == "zeroshot_clf":
        dataset = ZeroshotTestDataset(args.d_v, args.test_n, model)
        dl = DataLoader(dataset, batch_size=args.test_n, shuffle=True)
    elif stage == "support_clf":
        dataset = SupportTestDataset(args.d_v, args.test_n, model)
        dl = DataLoader(dataset, batch_size=args.test_n, shuffle=True)
    elif stage == "sum_clf":
        dataset = SumTestDataset(args.d_v, args.test_n, model)
        dl = DataLoader(dataset, batch_size=args.test_n, shuffle=True)
    return dl

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


def test_zeroshot_clf(test_loader, args):
    for r_a, r_b, r_c in test_loader:
        # TODO: I'm normalizing here. But I'm not normalizing for the other
        # evaluation methods. Is that okay?
        if args.normalize:
            r_a = F.normalize(r_a, p=2.0, dim=1)
            r_b = F.normalize(r_b, p=2.0, dim=1)
            r_c = F.normalize(r_c, p=2.0, dim=1)
        if args.loss_fn == "symile":
            # zeroshot_logits is a (batch_sz, batch_sz) matrix where each row i is
            # [ MIP(r_a[i], r_b[i], r_c[1]) MIP(r_a[i], r_b[i], r_c[2]) ... MIP(r_a[i], r_b[i], r_c[batch_sz]) ]
            # where MIP is the multilinear inner product.
            zeroshot_logits = (r_a * r_b) @ torch.t(r_c)
        elif args.loss_fn == "pairwise_infonce":
            # zeroshot_logits is a (batch_sz, batch_sz) matrix where each row i is
            # [ r_a[i]^T r_b[i] + r_b[i]^T r_c[1] + r_a[i]^T r_c[1] ... r_a[i]^T r_b[i] + r_b[i]^T r_c[batch_sz] + r_a[i]^T r_c[batch_sz] ]
            ab = torch.diagonal(r_a @ torch.t(r_b)).unsqueeze(dim=1) # (batch_sz, 1)
            ac_bc = (r_a @ torch.t(r_c)) + (r_b @ torch.t(r_c)) # (batch_sz, batch_sz)
            zeroshot_logits = ab + ac_bc
        # TODO: I'm supposed to scale by a temperature parameter.
        # But I'm not sure which temperature parameter to use?
        # The one last used by the model? See Section 3.1.2. of CLIP paper.
        logit_scale = 1.0
        zeroshot_logits = logit_scale * zeroshot_logits # (batch_sz, batch_sz)
        preds = torch.argmax(zeroshot_logits, dim=1)
        labels = torch.arange(r_a.shape[0])
        mean_acc = torch.where(preds == labels, 1, 0).sum() / len(labels)
        print("Mean accuracy: ", mean_acc)
        if args.wandb:
            wandb.log({"mean_acc": mean_acc})


def test_support_clf(test_loader, args, clf):
    for r_a, r_b, r_c, y in test_loader:
        if args.loss_fn == "symile":
            X = (r_a * r_b * r_c)
        elif args.loss_fn == "pairwise_infonce":
            if args.concat_infonce:
                X = torch.cat((r_a * r_b, r_b * r_c, r_a * r_c), dim=1)
            else:
                X = (r_a * r_b) + (r_b * r_c) + (r_a * r_c)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        mean_acc = clf.score(X_test, y_test)
        print("Mean accuracy: ", mean_acc)
        if args.wandb:
            wandb.log({"mean_acc": mean_acc})


def test_sum_clf(test_loader, args, clf):
    for r_a, r_b, y in test_loader:
        X = r_a * r_b
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        mean_acc = clf.score(X_test, y_test)
        print("Mean accuracy: ", mean_acc)
        if args.wandb:
            wandb.log({"mean_acc": mean_acc})


if __name__ == '__main__':
    if os.getenv('SINGULARITY_CONTAINER'):
        os.environ['WANDB_CACHE_DIR'] = '/scratch/as16583/python_cache/wandb/'
    args = parse_args()
    wandb_init(args)

    # pretraining
    print("\n\n\n...pretraining...\n")
    encoders = LinearEncoders(args.d_v, args.d_r)
    pt_loader = load_data(args, stage="pretrain")
    pt_val_loader = load_data(args, stage="pretrain_val")
    optimizer = torch.optim.AdamW(encoders.parameters(), lr=args.lr)
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    pretrain(pt_loader, pt_val_loader, encoders, loss_fn, optimizer, args)

    # evaluation
    for eval in args.evaluation.split(","):
        if eval == "zeroshot_clf":
            print("\n\n\n...evaluation: zero-shot classification...\n")
            test_loader = load_data(args, stage="zeroshot_clf", model=encoders)
            test_zeroshot_clf(test_loader, args)
        elif eval == "support_clf":
            print("\n\n\n...evaluation: in support classification...\n")
            test_loader = load_data(args, stage="support_clf", model=encoders)
            clf = LogisticRegression()
            test_support_clf(test_loader, args, clf)
        elif eval == "sum_clf":
            print("\n\n\n...evaluation: representation sum classification...\n")
            test_loader = load_data(args, stage="sum_clf", model=encoders)
            clf = LogisticRegression(multi_class="multinomial")
            test_sum_clf(test_loader, args, clf)