"""
Example script using Symile to train and test 8 simple linear encoders for the
following data generating procedure:

    v_a, v_b, v_c, v_d, v_e, v_f, v_g ~ Bernoulli(0.5)
    v_h = v_a XOR v_b XOR v_c XOR v_d XOR v_e XOR v_f XOR v_g

The zero-shot classification task is to predict whether v_a is 0 or 1 given the
remaining variables (v_b, v_c, ..., v_h).
"""
import numpy as np
from scipy.stats import bernoulli
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from symile import Symile, MIPSimilarity


class BinaryXORDataset(Dataset):
    """
    Generate n samples of data (v_a, v_b, ..., v_h) where
    v_a, v_b, ..., v_g ~ Bernoulli(0.5)
    v_h = v_a XOR v_b XOR v_c XOR ... XOR v_g
    """
    def __init__(self, n):
        self.v_a, self.v_b, self.v_c, self.v_d, self.v_e, self.v_f, self.v_g, self.v_h = self.generate_data(n)

    def generate_data(self, n):
        v_a = bernoulli.rvs(0.5, size=n)
        v_b = bernoulli.rvs(0.5, size=n)
        v_c = bernoulli.rvs(0.5, size=n)
        v_d = bernoulli.rvs(0.5, size=n)
        v_e = bernoulli.rvs(0.5, size=n)
        v_f = bernoulli.rvs(0.5, size=n)
        v_g = bernoulli.rvs(0.5, size=n)
        v_h = np.bitwise_xor.reduce([v_a, v_b, v_c, v_d, v_e, v_f, v_g])

        return [torch.from_numpy(v).float().unsqueeze(1) for v in [v_a, v_b, v_c, v_d, v_e, v_f, v_g, v_h]]

    def __len__(self):
        return len(self.v_a)

    def __getitem__(self, idx):
        return (self.v_a[idx], self.v_b[idx], self.v_c[idx], self.v_d[idx],
                self.v_e[idx], self.v_f[idx], self.v_g[idx], self.v_h[idx])


class Encoders(nn.Module):
    def __init__(self, d, logit_scale_init):
        super().__init__()
        self.f_a = nn.Linear(1, d, bias=True)
        self.f_b = nn.Linear(1, d, bias=True)
        self.f_c = nn.Linear(1, d, bias=True)
        self.f_d = nn.Linear(1, d, bias=True)
        self.f_e = nn.Linear(1, d, bias=True)
        self.f_f = nn.Linear(1, d, bias=True)
        self.f_g = nn.Linear(1, d, bias=True)
        self.f_h = nn.Linear(1, d, bias=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

    def forward(self, inputs):
        r_a = self.f_a(inputs[0])
        r_b = self.f_b(inputs[1])
        r_c = self.f_c(inputs[2])
        r_d = self.f_d(inputs[3])
        r_e = self.f_e(inputs[4])
        r_f = self.f_f(inputs[5])
        r_g = self.f_g(inputs[6])
        r_h = self.f_h(inputs[7])
        return r_a, r_b, r_c, r_d, r_e, r_f, r_g, r_h, self.logit_scale.exp()


def l2_normalize(vectors):
    """L2 normalize a list of 2D torch.Tensor vectors."""
    return [F.normalize(v, p=2.0, dim=1) for v in vectors]


def validation(val_loader, model, loss_fn, device):
    model.eval()
    total_val_loss = 0.0
    val_accuracies = []

    v_a_candidates = torch.tensor([[0.], [1.]]).to(device)
    r_a_candidates = model.f_a(v_a_candidates)
    r_a_candidates = l2_normalize([r_a_candidates])[0]

    similarity_fn = MIPSimilarity()

    with torch.no_grad():
        for batch in val_loader:
            batch = [b.to(device) for b in batch]
            *r_outputs, logit_scale_exp = model(batch)
            r_outputs = l2_normalize(r_outputs)

            # get val loss
            val_loss = loss_fn(r_outputs, logit_scale_exp)
            total_val_loss += val_loss.item()

            # get val accuracy
            _, r_b, r_c, r_d, r_e, r_f, r_g, r_h = r_outputs
            similarity_scores = similarity_fn(r_a_candidates, [r_b, r_c, r_d, r_e, r_f, r_g, r_h])
            similarity_scores = logit_scale_exp * similarity_scores

            preds = torch.argmax(similarity_scores, dim=1)
            labels = torch.squeeze(batch[0]).long()
            accuracies = (preds == labels).float().tolist()
            val_accuracies.extend(accuracies)

    mean_val_loss = total_val_loss / len(val_loader)
    mean_val_acc = sum(val_accuracies) / len(val_accuracies)

    return mean_val_loss, mean_val_acc


def train(train_loader, val_loader, args):
    device = args["device"]

    model = Encoders(args["d"], args["logit_scale_init"]).to(device)
    loss_fn = Symile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args["lr"])

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(args["epochs"]):
        # train step
        model.train()

        running_loss = 0.0

        for batch in train_loader:
            batch = [b.to(device) for b in batch]

            optimizer.zero_grad()

            *r_outputs, logit_scale_exp = model(batch)
            r_outputs = l2_normalize(r_outputs)

            loss = loss_fn(r_outputs, logit_scale_exp)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # validation step
        val_loss, val_acc = validation(val_loader, model, loss_fn, device)
        print(f"Epoch {epoch + 1} | Train Loss: {running_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        # save best model based on validation loss
        if val_loss < best_val_loss:
            print(f"Validation loss decreased ({best_val_loss:.4f} -> {val_loss:.4f}), saving model...")
            best_val_loss = val_loss
            best_model = model.state_dict()

    model.load_state_dict(best_model)

    return model


def test(test_loader, model, device):
    model.eval()
    test_accuracies = []

    # get candidate representations
    v_a_candidates = torch.tensor([[0.], [1.]]).to(device)
    r_a_candidates = model.f_a(v_a_candidates)
    r_a_candidates = l2_normalize([r_a_candidates])[0]

    similarity_fn = MIPSimilarity()

    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(device) for b in batch]
            *r_outputs, logit_scale_exp = model(batch)
            _, r_b, r_c, r_d, r_e, r_f, r_g, r_h = l2_normalize(r_outputs)

            similarity_scores = similarity_fn(r_a_candidates, [r_b, r_c, r_d, r_e, r_f, r_g, r_h])
            similarity_scores = logit_scale_exp * similarity_scores

            preds = torch.argmax(similarity_scores, dim=1)
            labels = torch.squeeze(batch[0]).long()
            accuracies = (preds == labels).float().tolist()
            test_accuracies.extend(accuracies)

    mean_acc = sum(test_accuracies) / len(test_accuracies)
    print(f"Test Accuracy: {mean_acc:.4f}")


if __name__ == "__main__":
    ### hyperparameters ###
    args = {"d": 16,
            "epochs": 50,
            "logit_scale_init": -0.3,
            "lr": 0.1,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}

    ### datasets ###
    train_n = 10000
    val_n = 1000
    test_n = 5000
    batch_sz = 1000

    dataset = BinaryXORDataset(train_n + val_n + test_n)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_n, val_n, test_n])

    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)

    ### training ###
    best_model = train(train_loader, val_loader, args)

    ### testing ###
    test(test_loader, best_model, args["device"])