"""
Example script demonstrating how to train Symile with missing modalities. The
data is generated as follows:

    v_a, v_b ~ Bernoulli(0.5)
    v_c = v_a XOR v_b

To simulate missingness in the data, for each vector v_a, v_b, v_c in the train
and val sets, values are randomly set to 0.5 with probability `args.missingness_prob`.

The zero-shot classification task is to predict whether v_a is 0 or 1 given the
remaining variables (v_b, v_c).
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
    Generate n samples of data (v_a, v_b, v_c) where
    v_a, v_b ~ Bernoulli(0.5)
    v_c = v_a XOR v_b

    If apply_missingness is True and missingness_prob > 0, then for each vector
    v_a, v_b, v_c, values are randomly set to 0.5 with probability missingness_prob.
    """
    def __init__(self, n, missingness_prob=0.0, apply_missingness=False):
        self.v_a, self.v_b, self.v_c = self.generate_data(n)

        # initialize missingness indicators
        self.m_a = torch.zeros_like(self.v_a)
        self.m_b = torch.zeros_like(self.v_b)
        self.m_c = torch.zeros_like(self.v_c)

        if apply_missingness and missingness_prob > 0:
            m_a = bernoulli.rvs(missingness_prob, size=n)
            m_b = bernoulli.rvs(missingness_prob, size=n)
            m_c = bernoulli.rvs(missingness_prob, size=n)

            self.m_a = torch.from_numpy(m_a).float().unsqueeze(1)
            self.m_b = torch.from_numpy(m_b).float().unsqueeze(1)
            self.m_c = torch.from_numpy(m_c).float().unsqueeze(1)

            # apply missingness by setting values to 0.5 where missing
            self.v_a = torch.where(self.m_a == 1, torch.tensor(0.5), self.v_a)
            self.v_b = torch.where(self.m_b == 1, torch.tensor(0.5), self.v_b)
            self.v_c = torch.where(self.m_c == 1, torch.tensor(0.5), self.v_c)

    def generate_data(self, n):
        v_a = bernoulli.rvs(0.5, size=n)
        v_b = bernoulli.rvs(0.5, size=n)
        v_c = np.bitwise_xor.reduce([v_a, v_b])

        return [torch.from_numpy(v).float().unsqueeze(1) for v in [v_a, v_b, v_c]]

    def __len__(self):
        return len(self.v_a)

    def __getitem__(self, idx):
        return (self.v_a[idx], self.v_b[idx], self.v_c[idx],
                self.m_a[idx], self.m_b[idx], self.m_c[idx])


class Encoders(nn.Module):
    def __init__(self, d, logit_scale_init):
        super().__init__()
        self.f_a = nn.Linear(2, d, bias=True)
        self.f_b = nn.Linear(2, d, bias=True)
        self.f_c = nn.Linear(2, d, bias=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * logit_scale_init)

    def forward(self, inputs):
        v_a, v_b, v_c, m_a, m_b, m_c = inputs

        input_a = torch.cat([v_a, m_a], dim=1)
        input_b = torch.cat([v_b, m_b], dim=1)
        input_c = torch.cat([v_c, m_c], dim=1)

        r_a = self.f_a(input_a)
        r_b = self.f_b(input_b)
        r_c = self.f_c(input_c)
        return r_a, r_b, r_c, self.logit_scale.exp()


def l2_normalize(vectors):
    """L2 normalize a list of 2D torch.Tensor vectors."""
    return [F.normalize(v, p=2.0, dim=1) for v in vectors]


def validation(val_loader, model, loss_fn, device):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = [b.to(device) for b in batch]
            *r_outputs, logit_scale_exp = model(batch)
            r_outputs = l2_normalize(r_outputs)

            val_loss = loss_fn(r_outputs, logit_scale_exp)
            total_val_loss += val_loss.item()

    mean_val_loss = total_val_loss / len(val_loader)

    return mean_val_loss


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
        val_loss = validation(val_loader, model, loss_fn, device)
        print(f"Epoch {epoch + 1} | Train Loss: {running_loss / len(train_loader):.4f} | "
              f"Val Loss: {val_loss:.4f}")

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
    m_a_candidates = torch.zeros_like(v_a_candidates).to(device) # both values are observed (m=0)
    input_a_candidates = torch.cat([v_a_candidates, m_a_candidates], dim=1)

    r_a_candidates = model.f_a(input_a_candidates)
    r_a_candidates = l2_normalize([r_a_candidates])[0]

    similarity_fn = MIPSimilarity()

    with torch.no_grad():
        for batch in test_loader:
            batch = [b.to(device) for b in batch]
            *r_outputs, logit_scale_exp = model(batch)
            _, r_b, r_c = l2_normalize(r_outputs)

            similarity_scores = similarity_fn(r_a_candidates, [r_b, r_c])
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
            "missingness_prob": 0.2,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")}

    ### datasets ###
    train_n = 10000
    val_n = 1000
    test_n = 5000
    batch_sz = 1000

    # create datasets with missingness for train/val but not test
    train_val_dataset = BinaryXORDataset(train_n + val_n,
                                         missingness_prob=args["missingness_prob"],
                                         apply_missingness=True)
    train_dataset, val_dataset = random_split(train_val_dataset, [train_n, val_n])

    test_dataset = BinaryXORDataset(test_n,
                                    missingness_prob=0.0,
                                    apply_missingness=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)

    ### training ###
    best_model = train(train_loader, val_loader, args)

    ### testing ###
    test(test_loader, best_model, args["device"])