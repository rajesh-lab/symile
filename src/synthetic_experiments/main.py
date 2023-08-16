"""
Experiment to demonstrate performance of SYMILE on synthetic datasets.
"""
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
try:
    import wandb
except ImportError:
    wandb = None

from datasets import PretrainingDataset, SumTestDataset, \
                     SupportTestDataset, ZeroshotTestDataset
from losses import pairwise_infonce, symile
from models import LinearEncoders
from params import parse_args
from utils import l2_normalize, seed_all, wandb_init


def load_data(args, stage="pretrain", model=None):
    if stage == "pretrain":
        dataset = PretrainingDataset(args.d_v, args.pretrain_n)
        batch_sz = args.pretrain_n if args.use_full_dataset else args.batch_sz_pt
        dl = DataLoader(dataset, batch_size=batch_sz, shuffle=True)
    elif stage == "pretrain_val":
        dataset = PretrainingDataset(args.d_v, args.pretrain_val_n)
        dl = DataLoader(dataset, batch_size=args.pretrain_val_n, shuffle=True)
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

def pretrain(args, model):
    pt_loader = load_data(args, stage="pretrain")
    pt_val_loader = load_data(args, stage="pretrain_val")
    loss_fn = symile if args.loss_fn == "symile" else pairwise_infonce
    optimizer = torch.optim.AdamW(encoders.parameters(), lr=args.lr)

    model.train()
    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(args.pt_epochs):
        for v_a, v_b, v_c in pt_loader:
            r_a, r_b, r_c, logit_scale_exp = model(v_a, v_b, v_c)
            assert r_c is not None, "r_c must be defined for pretraining."
            loss = loss_fn(r_a, r_b, r_c, logit_scale_exp, args.normalize)
            loss.backward()
            optimizer.step()
            if args.wandb:
                wandb.log({"pretrain_loss": loss,
                           "logit_scale_exp": logit_scale_exp})
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


def get_query_representations(encoders, normalize):
    """
    Returns representations for the four query vectors [0,0], [0,1], [1,0], [1,1].

    Args:
        encoders (models.LinearEncoders): trained encoders.
    Returns:
        q (torch.Tensor): query representations of size (4, d_r) where
                          q[0] = f([0,0]), q[1] = f([0,1]),
                          q[2] = f([1,0]), q[3] = f([1,1]).
    """
    encoders.eval()
    with torch.no_grad():
        q_00 = encoders.f_c(torch.Tensor([0,0]))
        q_01 = encoders.f_c(torch.Tensor([0,1]))
        q_10 = encoders.f_c(torch.Tensor([1,0]))
        q_11 = encoders.f_c(torch.Tensor([1,1]))
    assert torch.ne(q_00, q_01).any() and \
           torch.ne(q_00, q_10).any() and \
           torch.ne(q_00, q_11).any() and \
           torch.ne(q_01, q_10).any() and \
           torch.ne(q_01, q_11).any() and \
           torch.ne(q_10, q_11).any(), \
           "q_00, q_01, q_10, q_11 must all be different."
    q = torch.cat((torch.unsqueeze(q_00, 0), torch.unsqueeze(q_01, 0),
                   torch.unsqueeze(q_10, 0), torch.unsqueeze(q_11, 0)), dim=0)
    if normalize:
        [q] = l2_normalize([q])
    return q


def test_zeroshot_clf(args, encoders):
    test_loader = load_data(args, stage="zeroshot_clf", model=encoders)
    q = get_query_representations(encoders, args.normalize)

    for r_a, r_b, r_c in test_loader:
        if args.normalize:
            r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])

        # get predictions
        if args.loss_fn == "symile":
            # logits is a (batch_sz, 4) matrix where each row i is
            # [ MIP(r_a[i], r_b[i], q[0]) ... MIP(r_a[i], r_b[i], q[3]) ]
            # where MIP is the multilinear inner product.
            logits = (r_a * r_b) @ torch.t(q)
        elif args.loss_fn == "pairwise_infonce":
            # logits is a (batch_sz, 4) matrix where each row i is
            # [ r_a[i]^T r_b[i] + r_b[i]^T q[0] + r_a[i]^T q[0] ...
            #   r_a[i]^T r_b[i] + r_b[i]^T q[3] + r_a[i]^T q[3] ]
            ab = torch.diagonal(r_a @ torch.t(r_b)).unsqueeze(dim=1) # (batch_sz, 1)
            logits = ab + (r_b @ torch.t(q)) + (r_a @ torch.t(q))
        if args.use_logit_scale_eval:
            logits = encoders.logit_scale.exp().item() * logits
        preds = torch.argmax(logits, dim=1)

        # get labels
        def _get_label(r):
            return torch.argmax(torch.where(r == q, 1, 0).sum(dim=1))
        labels = torch.vmap(_get_label)(r_c)

        mean_acc = torch.where(preds == labels, 1, 0).sum() / len(labels)
        print("Mean accuracy: ", mean_acc)
        if args.wandb:
            wandb.log({"mean_acc": mean_acc})


def test_support_clf(args, encoders):
    test_loader = load_data(args, stage="support_clf", model=encoders)
    clf = LogisticRegression()

    for r_a, r_b, r_c, y in test_loader:
        if args.normalize:
            r_a, r_b, r_c = l2_normalize([r_a, r_b, r_c])
        if args.loss_fn == "symile":
            X = r_a * r_b * r_c
        elif args.loss_fn == "pairwise_infonce":
            if args.concat_infonce:
                X = torch.cat((r_a * r_b, r_b * r_c, r_a * r_c), dim=1)
            else:
                X = (r_a * r_b) + (r_b * r_c) + (r_a * r_c)
        if args.use_logit_scale_eval:
            X = encoders.logit_scale.exp().item() * X
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf.fit(X_train, y_train)
        mean_acc = clf.score(X_test, y_test)
        print("Mean accuracy: ", mean_acc)
        if args.wandb:
            wandb.log({"mean_acc": mean_acc})


def test_sum_clf(args, encoders):
    test_loader = load_data(args, stage="sum_clf", model=encoders)
    clf = LogisticRegression(multi_class="multinomial")

    for r_a, r_b, y in test_loader:
        if args.normalize:
            r_a, r_b = l2_normalize([r_a, r_b])
        X = r_a * r_b

        if args.use_logit_scale_eval:
            X = encoders.logit_scale.exp().item() * X

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
    if args.use_seed:
        seed_all(args.seed)

    # pretraining
    print("\n\n\n...pretraining...\n")
    encoders = LinearEncoders(args.d_v, args.d_r, args.logit_scale_init, args.hardcode_encoders)
    pretrain(args, encoders)

    # evaluation
    if eval == "zeroshot_clf":
        print("\n\n\n...evaluation: zero-shot classification...\n")
        test_zeroshot_clf(args, encoders)
    elif eval == "support_clf":
        print("\n\n\n...evaluation: in support classification...\n")
        test_support_clf(args, encoders)
    elif eval == "sum_clf":
        print("\n\n\n...evaluation: representation sum classification...\n")
        test_sum_clf(args, encoders)